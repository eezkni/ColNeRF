"""
Full evaluation script, including PSNR+SSIM evaluation with multi-GPU support.

python eval.py --gpu_id=<gpu list> -n <expname> -c <conf> -D /home/group/data/chairs -F srn
"""
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import torch
import numpy as np
import imageio
# import skimage.measure
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

import util
from data import get_split_dataset, load_llff_data
from model import make_model
from render import NeRFRenderer
import cv2
import tqdm
import ipdb
import warnings
import random

#  from pytorch_memlab import set_target_gpu
#  set_target_gpu(9)


def extra_args(parser):
    parser.add_argument(
        "--eval_view_list", type=str, default=None, help="Path to eval view list"
    )
    parser.add_argument("--coarse", action="store_true", help="Coarse network as fine")
    parser.add_argument(
        "--no_compare_gt",
        action="store_true",
        help="Skip GT comparison (metric won't be computed) and only render images",
    )
    parser.add_argument(
        "--multicat",
        action="store_true",
        help="Prepend category id to object id. Specify if model fits multiple categories.",
    )
    parser.add_argument(
        "--viewlist",
        "-L",
        type=str,
        default="",
        help="Path to source view list e.g. src_dvr.txt; if specified, overrides source/P",
    )

    parser.add_argument(
        "--output",
        "-O",
        type=str,
        default="eval",
        help="If specified, saves generated images to directory",
    )
    parser.add_argument(
        "--include_src", action="store_true", help="Include source views in calculation"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Video scale relative to input size"
    )
    parser.add_argument("--write_depth", action="store_true", help="Write depth image")
    parser.add_argument(
        "--write_compare", action="store_true", help="Write GT comparison image"
    )
    return parser


args, conf = util.args.parse_args(
    extra_args, default_conf="conf/resnet_fine_mv.conf", default_expname="shapenet",
)
args.resume = True

device = util.get_cuda(args.gpu_id[0])

net = make_model(conf["model"], data_format=args.dataset_format, nviews=int(args.nviews[0])).to(device=device)
renderer = NeRFRenderer.from_conf(
            conf["renderer"], lindisp=False, eval_batch_size=args.ray_batch_size
        ).to(device=device)

## compute scenes
scenes = [f for f in os.listdir(args.datadir)]
total_scenes = len(scenes)

output_dir = args.output.strip()
has_output = len(output_dir) > 0

total_psnr = 0.0
total_ssim = 0.0
cnt = 0

if has_output:
    finish_path = os.path.join(output_dir, "finish.txt")
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(finish_path):
        with open(finish_path, "r") as f:
            lines = [x.strip().split() for x in f.readlines()]
        lines = [x for x in lines if len(x) == 4]
        finished = set([x[0] for x in lines])
        total_psnr = sum((float(x[1]) for x in lines))
        total_ssim = sum((float(x[2]) for x in lines))
        cnt = sum((int(x[3]) for x in lines))
        if cnt > 0:
            print("resume psnr", total_psnr / cnt, "ssim", total_ssim / cnt)
        else:
            total_psnr = 0.0
            total_ssim = 0.0
    else:
        finished = set()

    finish_file = open(finish_path, "a", buffering=1)
    print("Writing images to", output_dir)
    
base_checkpoints_path = args.checkpoints_path
args.checkpoints_path = os.path.join(base_checkpoints_path, args.name)
base_datadir = args.datadir
    
with torch.no_grad():
    for i, scene in enumerate(scenes):
        print(
            "Scene",
            i,
            "OF",
            total_scenes,
            "PROGRESS",
            i / total_scenes * 100.0,
            "%",
        )
        basedir = os.path.join(args.logs_path, args.name)
        args.name = scene
        args.datadir = os.path.join(base_datadir, args.name)
        net.load_weights(args)
        if args.coarse:
            net.mlp_fine = None

        if renderer.n_coarse < 64:
            # Ensure decent sampling resolution
            renderer.n_coarse = 64
        if args.coarse:
            renderer.n_coarse = 64
            renderer.n_fine = 128
            renderer.using_fine = True

        render_par = renderer.bind_parallel(net, args.gpu_id, simple_output=True).eval()

        K = None
        ori_images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                        recenter=True, bd_factor=.75,
                                                                        spherify=args.spherify)
        ori_images = (torch.from_numpy(ori_images) - 0.5) / 0.5
        ori_images = ori_images.permute(0, 3, 1, 2) # (NV, 3, H, W)
        images = ori_images.to(device=device)  # (NV, 3, H, W)
        # dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir)
        hwf = poses[0,:3,-1] #shape:3
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold] # 0, 8, 16

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)]) # other

        print('DEFINING BOUNDS')

        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.
        print('NEAR FAR', near, far)
        
        z_near = torch.tensor(near).float()
        z_far = torch.tensor(far).float()

        # Cast intrinsics to right types
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        train_images = images[i_train]
        test_images = images[i_test]

        if K is None:
            K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])

        # Create log dir and copy the config file
        expname = args.name
        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        f = os.path.join(basedir, expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        
        c = None
        poses = torch.from_numpy(poses).to(device=device)   # (NV, 4, 4)
        focal = torch.tensor(focal).unsqueeze(0)
        if isinstance(focal, float):
            focal = torch.tensor(focal, dtype=torch.float32)
        focal = focal[None].repeat(1,2)
        
        NV, _, H, W = images.shape

        use_source_lut = len(args.viewlist) > 0
        if use_source_lut:
            print("Using views from list", args.viewlist)
            with open(args.viewlist, "r") as f:
                tmp = [x.strip().split() for x in f.readlines()]
            source_lut = {
                x[0] + "/" + x[1]: torch.tensor(list(map(int, x[2:])), dtype=torch.long)
                for x in tmp
            }
        else:
            source = random.sample(list(i_train), int(args.nviews[0]))

        eval_views = i_test
        target_view_mask = torch.zeros(NV, dtype=torch.bool)
        target_view_mask[eval_views] = 1

        target_view_mask_init = target_view_mask

        all_rays = None
        rays_spl = []

        src_view_mask = None
        total_objs = test_images.shape[0]

        if args.scale != 1.0:
            Ht = int(H * args.scale)
            Wt = int(W * args.scale)
            if abs(Ht / args.scale - H) > 1e-10 or abs(Wt / args.scale - W) > 1e-10:
                warnings.warn(
                    "Inexact scaling, please check {} times ({}, {}) is integral".format(
                        args.scale, H, W
                    )
                )
            H, W = Ht, Wt
            
        cat_name = os.path.basename(os.path.dirname(args.datadir))
        obj_name = cat_name + "_" + scene if args.multicat else scene

        if all_rays is None or use_source_lut:
            if use_source_lut:
                obj_id = cat_name + "/" + scene
                source = source_lut[obj_id]

            NS = len(source)
            src_view_mask = torch.zeros(NV, dtype=torch.bool)
            src_view_mask[source] = 1

            c = torch.tensor((0.5*W, 0.5*H)).unsqueeze(0)
            if c is not None:
                c = c[0].to(device=device).unsqueeze(0)

            src_poses = poses[src_view_mask].to(device=device)  # (NS, 4, 4)

            target_view_mask = target_view_mask_init.clone()
            if not args.include_src:
                target_view_mask *= ~src_view_mask

            novel_view_idxs = target_view_mask.nonzero(as_tuple=False).reshape(-1)

            poses = poses[target_view_mask]  # (NV[-NS], 4, 4)

            all_rays = (
                util.gen_rays(
                    poses.reshape(-1, 3, 4),
                    W,
                    H,
                    focal * args.scale,
                    z_near,
                    z_far,
                    c=c * args.scale if c is not None else None,
                )
                .reshape(-1, 8)
                .to(device=device)
            )  # ((NV[-NS])*H*W, 8)

            poses = None
            focal = focal.to(device=device)

        rays_spl = torch.split(all_rays, args.ray_batch_size, dim=0)  # Creates views

        n_gen_views = len(novel_view_idxs)

        net.encode(
            images[src_view_mask].to(device=device).unsqueeze(0),
            src_poses.unsqueeze(0),
            focal,
            c=c,
        )

        all_rgb, all_depth = [], []
        for rays in tqdm.tqdm(rays_spl):
            rgb, depth = render_par(rays[None])
            rgb = rgb[0].cpu()
            depth = depth[0].cpu()
            all_rgb.append(rgb)
            all_depth.append(depth)

        all_rgb = torch.cat(all_rgb, dim=0)
        all_depth = torch.cat(all_depth, dim=0)
        all_depth = (all_depth - z_near) / (z_far - z_near)
        all_depth = all_depth.reshape(n_gen_views, H, W).numpy()

        all_rgb = torch.clamp(
            all_rgb.reshape(n_gen_views, H, W, 3), 0.0, 1.0
        ).numpy()  # (NV-NS, H, W, 3)
        if has_output:
            obj_out_dir = os.path.join(output_dir, obj_name)
            os.makedirs(obj_out_dir, exist_ok=True)
            depth_max = np.nanmax(all_depth)
            depth_min = np.nanmin(all_depth)
            # all_depth_r = 1 - all_depth
            print("depth_max:", depth_max)
            print("depth_min:", depth_min)
            for i in range(n_gen_views):
                out_file = os.path.join(
                    obj_out_dir, "{:06}.png".format(novel_view_idxs[i].item())
                )
                imageio.imwrite(out_file, (all_rgb[i] * 255).astype(np.uint8))

                if args.write_depth:
                    out_depth_file = os.path.join(
                        obj_out_dir, "{:06}_depth.exr".format(novel_view_idxs[i].item())
                    )
                    out_depth_norm_file = os.path.join(
                        obj_out_dir,
                        "{:06}_depth_norm.png".format(novel_view_idxs[i].item()),
                    )
                    depth_cmap_norm = (all_depth[i] - depth_min) / (depth_max - depth_min) * 255
                    depth_cmap_norm = depth_cmap_norm.astype(np.uint8)
                    # depth_cmap_norm = util.cmap(all_depth[i])
                    # print("depth:", all_depth[i])
                    cv2.imwrite(out_depth_file, all_depth[i])
                    imageio.imwrite(out_depth_norm_file, depth_cmap_norm)

        curr_ssim = 0.0
        curr_psnr = 0.0
        if not args.no_compare_gt:
            images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)
            images_gt = images_0to1[target_view_mask]
            rgb_gt_all = (
                images_gt.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            )  # (NV-NS, H, W, 3)
            for view_idx in range(n_gen_views):
                ssim = structural_similarity(
                    all_rgb[view_idx],
                    rgb_gt_all[view_idx],
                    multichannel=True,
                    data_range=1,
                )
                psnr = peak_signal_noise_ratio(
                    all_rgb[view_idx], rgb_gt_all[view_idx], data_range=1
                )
                curr_ssim += ssim
                curr_psnr += psnr

                if args.write_compare:
                    out_file = os.path.join(
                        obj_out_dir,
                        "{:06}_compare.png".format(novel_view_idxs[view_idx].item()),
                    )
                    out_im = np.hstack((all_rgb[view_idx], rgb_gt_all[view_idx]))
                    imageio.imwrite(out_file, (out_im * 255).astype(np.uint8))
        curr_psnr /= n_gen_views
        curr_ssim /= n_gen_views
        curr_cnt = 1
        total_psnr += curr_psnr
        total_ssim += curr_ssim
        cnt += curr_cnt
        if not args.no_compare_gt:
            print(
                "curr psnr",
                curr_psnr,
                "ssim",
                curr_ssim,
                "running psnr",
                total_psnr / cnt,
                "running ssim",
                total_ssim / cnt,
            )
        finish_file.write(
            "{} {} {} {}\n".format(obj_name, curr_psnr, curr_ssim, curr_cnt)
        )
    print("final psnr", total_psnr / cnt, "ssim", total_ssim / cnt)