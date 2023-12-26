import os.path
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tqdm
import warnings
import random
import torch.backends.cudnn as cudnn

def _init_fn(worker_id): 
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)
    torch.cuda.manual_seed(seed + worker_id)
    torch.cuda.manual_seed_all(seed + worker_id)

class Trainer:
    def __init__(self, net, train_dataset, test_dataset, args, conf, device=None):
        self.args = args
        self.net = net
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=False,
            worker_init_fn=_init_fn
        )
        self.test_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=min(args.batch_size, 16),
            shuffle=True,
            num_workers=4,
            pin_memory=False,
            worker_init_fn=_init_fn
        )

        self.num_total_batches = len(self.train_dataset)
        self.exp_name = args.name
        self.save_interval = conf.get_int("save_interval")
        self.print_interval = conf.get_int("print_interval")
        self.vis_interval = conf.get_int("vis_interval")
        self.eval_interval = conf.get_int("eval_interval")
        self.perceptual_interval = conf.get_int("perceptual_interval")
        self.num_epoch_repeats = conf.get_int("num_epoch_repeats", 1)
        self.num_epochs = args.epochs
        self.accu_grad = conf.get_int("accu_grad", 1)
        self.summary_path = os.path.join(args.logs_path, args.name)
        self.writer = SummaryWriter(self.summary_path)

        self.fixed_test = hasattr(args, "fixed_test") and args.fixed_test

        os.makedirs(self.summary_path, exist_ok=True)

        # Currently only Adam supported
        self.optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        if args.gamma != 1.0:
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optim, gamma=args.gamma
            )
            # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optim, T_0=5, T_mult=2, eta_min=1e-5, last_epoch=-1, verbose=False)
        else:
            self.lr_scheduler = None

        # Load weights
        self.managed_weight_saving = hasattr(net, "load_weights")
        if self.managed_weight_saving:
            net.load_weights(self.args)
        self.iter_state_path = "%s/%s/_iter" % (
            self.args.checkpoints_path,
            self.args.name,
        )
        self.optim_state_path = "%s/%s/_optim" % (
            self.args.checkpoints_path,
            self.args.name,
        )
        self.lrsched_state_path = "%s/%s/_lrsched" % (
            self.args.checkpoints_path,
            self.args.name,
        )
        self.default_net_state_path = "%s/%s/net" % (
            self.args.checkpoints_path,
            self.args.name,
        )
        if not os.path.exists(self.default_net_state_path):
            os.makedirs(self.default_net_state_path, exist_ok=True)
        self.start_iter_id = 0
        if args.resume:
            if os.path.exists(self.optim_state_path):
                try:
                    self.optim.load_state_dict(
                        torch.load(self.optim_state_path, map_location=device)
                    )
                except:
                    warnings.warn(
                        "Failed to load optimizer state at", self.optim_state_path
                    )
            if self.lr_scheduler is not None and os.path.exists(
                self.lrsched_state_path
            ):
                self.lr_scheduler.load_state_dict(
                    torch.load(self.lrsched_state_path, map_location=device)
                )
            if os.path.exists(self.iter_state_path):
                self.start_iter_id = torch.load(
                    self.iter_state_path, map_location=device
                )["iter"]
            if not self.managed_weight_saving and os.path.exists(
                self.default_net_state_path
            ):
                net.load_state_dict(
                    torch.load(self.default_net_state_path, map_location=device)
                )

        self.visual_path = os.path.join(self.args.visual_path, self.args.name)
        self.conf = conf

    def post_batch(self, epoch, batch):
        """
        Ran after each batch
        """
        pass

    def extra_save_state(self):
        """
        Ran at each save step for saving extra state
        """
        pass

    def train_step(self, data, global_step, ):
        """
        Training step
        """
        raise NotImplementedError()

    def eval_step(self, data, global_step, ):
        """
        Evaluation step
        """
        raise NotImplementedError()

    def vis_step(self, data, global_step):
        """
        Visualization step
        """
        return None, None

    def start(self):
        def fmt_loss_str(losses):
            return "loss " + (" ".join(k + ":" + str(losses[k]) for k in losses))

        def data_loop(dl):
            """
            Loop an iterable infinitely
            """
            while True:
                for x in iter(dl):
                    yield x

        test_data_iter = data_loop(self.test_data_loader)
        
        self.best_test_loss = 1e6
        self.best_epoch = 0
        self.best_batch = 0

        step_id = self.start_iter_id

        total = self.num_epochs*self.num_epoch_repeats*len(self.train_data_loader)
        allProgress = tqdm.tqdm(total=total, desc='Train epoches', initial=0)
        for epoch in range(self.num_epochs):
            allProgress.write("--------- no. of steps for each epoch: {}".format(len(self.train_data_loader)))
            self.writer.add_scalar(
                "lr", self.optim.param_groups[0]["lr"], global_step=step_id
            )

            batch = 0
            for _ in range(self.num_epoch_repeats):
                for data in self.train_data_loader:
                    losses = self.train_step(data, global_step=step_id, )
                    loss_str = fmt_loss_str(losses)
                    if batch % self.print_interval == 0:
                        
                        allProgress.write(
                            "E {}, B {}, {}, Lr {}".format(epoch, batch, loss_str, self.optim.param_groups[0]["lr"])
                        )
                    
                    if batch % self.eval_interval == 0:
                        test_data = next(test_data_iter)
                        self.net.eval()
                        with torch.no_grad():
                            test_losses = self.eval_step(test_data, global_step=step_id, )
                            if self.best_test_loss > test_losses["rf"]:
                                self.best_test_loss = test_losses["rf"]
                                self.best_epoch = epoch
                                self.best_batch = batch
                                self.net.save_best_weights(self.args)
                        self.net.train()
                        self.writer.add_scalars("train", losses, global_step=step_id)
                        self.writer.add_scalars(
                            "test", test_losses, global_step=step_id
                        )
                        allProgress.write("*** Eval: E {}, B {}, test_loss {:2.6f}, best_test_loss {:2.6f}, best_epoch {}, best_step {}".format(epoch, batch, test_losses["rf"], self.best_test_loss, self.best_epoch, self.best_batch))

                    if step_id % self.save_interval == 0 and (epoch > 0 or batch > 0):
                        allProgress.write("saving")
                        if self.managed_weight_saving:
                            self.net.save_weights(self.args)
                        save_name = 'net_weights_' + str(step_id + 1)
                        save_path = os.path.join(self.default_net_state_path, save_name)
                        torch.save(
                            self.net.state_dict(), save_path
                        )
                        print("Saved model weights at step", step_id, "to", save_path)
                        torch.save(self.optim.state_dict(), self.optim_state_path)
                        if self.lr_scheduler is not None:
                            torch.save(
                                self.lr_scheduler.state_dict(), self.lrsched_state_path
                            )
                        torch.save({"iter": step_id + 1}, self.iter_state_path)
                        self.extra_save_state()

                    if batch % self.vis_interval == 0:
                        print("generating visualization")
                        if self.fixed_test:
                            test_data = next(iter(self.test_data_loader))
                        else:
                            test_data = next(test_data_iter)
                        self.net.eval()
                        with torch.no_grad():
                            vis, vis_vals = self.vis_step(
                                test_data, global_step=step_id
                            )
                        if vis_vals is not None:
                            self.writer.add_scalars(
                                "vis", vis_vals, global_step=step_id
                            )
                        self.net.train()
                        if vis is not None:
                            import imageio

                            vis_u8 = (vis * 255).astype(np.uint8)
                            imageio.imwrite(
                                os.path.join(
                                    self.visual_path,
                                    "{:04}_{:04}_vis.png".format(epoch, batch),
                                ),
                                vis_u8,
                            )

                    if (
                        batch == self.num_total_batches - 1
                        or batch % self.accu_grad == self.accu_grad - 1
                    ):
                        self.optim.step()
                        self.optim.zero_grad()

                    self.post_batch(epoch, batch)
                    step_id += 1
                    batch += 1
                    allProgress.update(1)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
