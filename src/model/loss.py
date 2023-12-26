import torch
import torchvision.models as models
import torch.nn.functional as F


class AlphaLossNV2(torch.nn.Module):
    """
    Implement Neural Volumes alpha loss 2
    """

    def __init__(self, lambda_alpha, clamp_alpha, init_epoch, force_opaque=False):
        super().__init__()
        self.lambda_alpha = lambda_alpha
        self.clamp_alpha = clamp_alpha
        self.init_epoch = init_epoch
        self.force_opaque = force_opaque
        if force_opaque:
            self.bceloss = torch.nn.BCELoss()
        self.register_buffer(
            "epoch", torch.tensor(0, dtype=torch.long), persistent=True
        )

    def sched_step(self, num=1):
        self.epoch += num

    def forward(self, alpha_fine):
        if self.lambda_alpha > 0.0 and self.epoch.item() >= self.init_epoch:
            alpha_fine = torch.clamp(alpha_fine, 0.01, 0.99)
            if self.force_opaque:
                alpha_loss = self.lambda_alpha * self.bceloss(
                    alpha_fine, torch.ones_like(alpha_fine)
                )
            else:
                alpha_loss = torch.log(alpha_fine) + torch.log(1.0 - alpha_fine)
                alpha_loss = torch.clamp_min(alpha_loss, -self.clamp_alpha)
                alpha_loss = self.lambda_alpha * alpha_loss.mean()
        else:
            alpha_loss = torch.zeros(1, device=alpha_fine.device)
        return alpha_loss


def get_alpha_loss(conf):
    lambda_alpha = conf.get_float("lambda_alpha")
    clamp_alpha = conf.get_float("clamp_alpha")
    init_epoch = conf.get_int("init_epoch")
    force_opaque = conf.get_bool("force_opaque", False)

    return AlphaLossNV2(
        lambda_alpha, clamp_alpha, init_epoch, force_opaque=force_opaque
    )


class RGBWithUncertainty(torch.nn.Module):
    """Implement the uncertainty loss from Kendall '17"""

    def __init__(self, conf):
        super().__init__()
        self.element_loss = (
            torch.nn.L1Loss(reduction="none")
            if conf.get_bool("use_l1")
            else torch.nn.MSELoss(reduction="none")
        )

    def forward(self, outputs, targets, betas):
        """computes the error per output, weights each element by the log variance
        outputs is B x 3, targets is B x 3, betas is B"""
        weighted_element_err = (
            torch.mean(self.element_loss(outputs, targets), -1) / betas
        )
        return torch.mean(weighted_element_err) + torch.mean(torch.log(betas))


class RGBWithBackground(torch.nn.Module):
    """Implement the uncertainty loss from Kendall '17"""

    def __init__(self, conf):
        super().__init__()
        self.element_loss = (
            torch.nn.L1Loss(reduction="none")
            if conf.get_bool("use_l1")
            else torch.nn.MSELoss(reduction="none")
        )

    def forward(self, outputs, targets, lambda_bg):
        """If we're using background, then the color is color_fg + lambda_bg * color_bg.
        We want to weight the background rays less, while not putting all alpha on bg"""
        weighted_element_err = torch.mean(self.element_loss(outputs, targets), -1) / (
            1 + lambda_bg
        )
        return torch.mean(weighted_element_err) + torch.mean(torch.log(lambda_bg))


def get_rgb_loss(conf, coarse=True, using_bg=False, reduction="mean"):
    if conf.get_bool("use_uncertainty", False) and not coarse:
        print("using loss with uncertainty")
        return RGBWithUncertainty(conf)
    #     if using_bg:
    #         print("using loss with background")
    #         return RGBWithBackground(conf)
    print("using vanilla rgb loss")
    return (
        torch.nn.L1Loss(reduction=reduction)
        if conf.get_bool("use_l1")
        else torch.nn.MSELoss(reduction=reduction)
    )
    
class EntropyLoss:
    def __init__(self):
        super(EntropyLoss, self).__init__()
        # self.N_samples = args.N_rand
        self.type_ = 'log2'
        self.threshold = 0.1
        # self.computing_entropy_all = args.computing_entropy_all
        # self.smoothing = args.smoothing
        self.computing_ignore_smoothing = False
        # self.entropy_log_scaling = args.entropy_log_scaling
        # self.N_entropy = args.N_entropy 
        
        # if self.N_entropy ==0:
        #     self.computing_entropy_all = True

    def ray_zvals(self, sigma, acc):
        # if self.smoothing and self.computing_ignore_smoothing:
        #     N_smooth = sigma.size(0)//2
        #     acc = acc[:N_smooth]
        #     sigma = sigma[:N_smooth]
        # if not self.computing_entropy_all:
        #     acc = acc[self.N_samples:]
        #     sigma = sigma[self.N_samples:]
        ray_prob = sigma / (torch.sum(sigma,-1).unsqueeze(-1)+1e-10)
        entropy_ray = self.entropy(ray_prob)
        entropy_ray_loss = torch.sum(entropy_ray, -1).unsqueeze(-1)
        
        # masking no hitting poisition?
        mask = (acc>self.threshold).detach()
        entropy_ray_loss*= mask
        # if self.entropy_log_scaling:
        #     return torch.log(torch.mean(entropy_ray_loss) + 1e-10)
        return torch.mean(entropy_ray_loss)
    
    def entropy(self, prob):
        if self.type_ == 'log2':
            return -1*prob*torch.log2(prob+1e-10)
        elif self.type_ == '1-p':
            return prob*torch.log2(1-prob)
        
class SmoothingLoss:
    def __init__(self):
        super(SmoothingLoss, self).__init__()
    
        self.smoothing_activation = 'norm'
        self.criterion = torch.nn.KLDivLoss(reduction='batchmean')
        self.threshold = 0.1
        self.type_ = 'log2'
        self.depth_loss = torch.nn.L1Loss()
        # self.weight = 0.01
    
    def __call__(self, sigma, acc, depth_1, depth_2, weight):
            
        npoints = sigma.shape[-1]
        bs = sigma.shape[0]
        nrays = depth_1.shape[-1]
        sigma = sigma.reshape(-1, npoints)
        acc = acc.reshape(-1, 1)
        
        mask = (acc>self.threshold).float()
        # if torch.any(mask) == 0:
        #     print("mask!")
        depth_1 = depth_1.reshape(bs * nrays,)
        depth_2 = depth_2.reshape(bs * nrays,)
        
        ray_prob = sigma / (torch.sum(sigma,-1).unsqueeze(-1)+1e-10)
        entropy_ray = self.entropy(ray_prob)
        entropy_ray_loss = torch.sum(entropy_ray, -1).unsqueeze(-1)
        
        # masking no hitting poisition?
        entropy_ray_loss = mask * entropy_ray_loss
        depth_loss = self.depth_loss(depth_1,depth_2)
    
        loss = weight * torch.mean(entropy_ray_loss)+torch.mean(depth_loss)
    
        return loss
    
    def entropy(self, prob):
        if self.type_ == 'log2':
            return -1*prob*torch.log2(prob+1e-10)
        elif self.type_ == '1-p':
            return prob*torch.log2(1-prob)
        
    
class RGBSmooth:
    def __init__(self):
        super(RGBSmooth, self).__init__()
    
        self.smoothing_activation = 'norm'
        self.criterion = torch.nn.KLDivLoss(reduction='batchmean')
        self.threshold = 0.02
    
    def __call__(self, rgb_pseudo, rgbs):
        """
        :param: rgb_pseudo (B, K, 3+1)
        :rgbs: rgb_pseudo (B, K, 3)
        """
        npoints= rgb_pseudo.shape[1]
        mask = rgb_pseudo[..., -1].unsqueeze(-1).permute(0, 2, 1) # (B,1, K,)
        rgb_pseudo = rgb_pseudo[..., :-1].permute(0, 2, 1) # (B,3, K,)
        rgbs = rgbs.permute(0, 2, 1) # (B,3, K,)
        rgbs = rgbs * mask
        
        if self.smoothing_activation == 'softmax':
            p = F.softmax(rgbs, -1)
            q = F.softmax(rgb_pseudo, -1)
        elif self.smoothing_activation == 'norm':
            p = rgbs / (torch.sum(rgbs, -1,  keepdim=True) + 1e-8) + 1e-8
            q = rgb_pseudo / (torch.sum(rgb_pseudo, -1, keepdim=True) + 1e-8) +1e-8
        loss = self.criterion(p.log(), q)
        
        return torch.mean(loss)
    
# class RaySmoothLoss:
#     def __init__(self):
#         super(RaySmoothLoss, self).__init__()
#         self.losstype = 'l2'
        
#     def __call__(self, density, deltas, weighting=None):
#         if len(density.shape) != 3:
#             density = density.unsqueeze(0) # bs, Nray, Npoints
#         if len(deltas.shape) != 3:
#             deltas = deltas.unsqueeze(0) # bs, Nray, Npoints-1
#         Nrays = density.shape[1]
#         Npoints = density.shape[-1]
        
#         weights = torch.sum(deltas, dim=-1).unsqueeze(-1)/Npoints
        
#         weights = weights/(deltas + 1e-6)
#         weights = torch.clamp(weights, min=0, max=5)
        
#         d1 = density[..., :Npoints-1]
#         d2 = density[..., 1:]        
        
#         if self.losstype == 'l2':
#             loss = ((d1 - d2) * weights) ** 2
#         elif self.losstype == 'l1':
#             loss = torch.abs(d1 - d2)
#         else:
#             raise ValueError('Not supported losstype.')

#         loss = torch.mean(loss)
#         if weighting is not None:
#             loss = loss * weighting
#         return loss
        
class VGG19_relu(torch.nn.Module):
    def __init__(self, device='cuda'):
        super(VGG19_relu, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn = models.vgg19(pretrained=True)
        # cnn = models.vgg19()
        # cnn = getattr(models, 'vgg19')
        # cnn.load_state_dict(torch.load(os.path.join('./models/', 'vgg19-dcbb9e9d.pth')))
        cnn = cnn.to(device)
        features = cnn.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


class PerceptualLoss(torch.nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0], resize=False, criterion='l1', device='cuda'):
        super(PerceptualLoss, self).__init__()
        if criterion == 'l1':
            self.criterion =torch.nn.L1Loss()
        elif criterion == 'sl1':
            self.criterion = torch.nn.SmoothL1Loss()
        elif criterion == 'l2':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError('Loss [{}] is not implemented'.format(criterion))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.add_module('vgg', VGG19_relu(device))
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        self.weights = weights
        self.resize = resize
        self.transformer = torch.nn.functional.interpolate

    def __call__(self, x, y):
        if self.resize:
            x = self.transformer(x, mode='bicubic', size=(224, 224), align_corners=True)
            y = self.transformer(y, mode='bicubic', size=(224, 224), align_corners=True)
        
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x = (x - self.mean.to(x)) / self.std.to(x)
        y = (y - self.mean.to(y)) / self.std.to(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss = 0.0
        loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return loss
