import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import torch
import torch.nn.functional as F
import numpy as np
import imageio
import util
import warnings
from data import get_split_dataset, load_llff_data
from render import NeRFRenderer
from model import make_model
from scipy.interpolate import CubicSpline
import tqdm
import random


def extra_args(parser):
    parser.add_argument(
        "--subset", "-S", type=int, default=0, help="Subset in data to use"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of data to use train | val | test",
    )
    parser.add_argument(
        "--source",
        "-P",
        type=str,
        default="22 25 28",
        help="Source view(s) in image, in increasing order. -1 to do random",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=40,
        help="Number of video frames (rotated views)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=-10.0,
        help="Elevation angle (negative is above)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Video scale relative to input size"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.0,
        help="Distance of camera from origin, default is average of z_far, z_near of dataset (only for non-DTU)",
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS of video")
    return parser

args, conf = util.args.parse_args(extra_args)
args.resume = True

nv = args.nviews + "v"
os.makedirs(os.path.join(args.visual_path, args.name, nv), exist_ok=True)

device = util.get_cuda(args.gpu_id[0])
## dtu
if args.dataset_format == 'dvr_dtu':
    dset = get_split_dataset(
        args.dataset_format, args.datadir, want_split=args.split, training=False
    )

    data = dset[args.subset]
    data_path = data["path"]
    print("Data instance loaded:", data_path)

    images = data["images"]  # (NV, 3, H, W)
    ori_images = images
    poses = data["poses"]  # (NV, 4, 4)
    focal = data["focal"]
    if isinstance(focal, float):
        # Dataset implementations are not consistent about
        # returning float or scalar tensor in case of fx=fy
        focal = torch.tensor(focal, dtype=torch.float32)
    focal = focal[None]

    c = data.get("c")
    if c is not None:
        c = c.to(device=device).unsqueeze(0)
    # Get the distance from camera to origin
    z_near = dset.z_near
    z_far = dset.z_far
    
## llff    
else: 
    K = None
    ori_images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                    recenter=True, bd_factor=.75,
                                                                    spherify=args.spherify)
    ori_images = (torch.from_numpy(ori_images) - 0.5) / 0.5
    ori_images = ori_images.permute(0, 3, 1, 2)
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
    basedir = args.logs_path
    expname = args.name
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
            
    dset = None
    c = None
    poses = torch.from_numpy(poses).to(device=device)   # (NV, 4, 4)
    focal = torch.tensor(focal).unsqueeze(0)
    if isinstance(focal, float):
        focal = torch.tensor(focal, dtype=torch.float32)
    focal = focal[None].repeat(1,2)
    
NV, _, H, W = images.shape

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

net = make_model(conf["model"], data_format=args.dataset_format, nviews=int(args.nviews)).to(device=device)
net.load_weights(args)

renderer = NeRFRenderer.from_conf(
    conf["renderer"], lindisp=False, eval_batch_size=args.ray_batch_size,
).to(device=device)

render_par = renderer.bind_parallel(net, args.gpu_id, simple_output=True).eval()

print("Generating rays")

if dset is not None and hasattr(dset, "sub_format") and dset.sub_format == "dtu":
    print("Using DTU camera trajectory")
    # Use hard-coded pose interpolation from IDR for DTU

    t_in = np.array([0, 2, 3, 5, 6]).astype(np.float32)
    pose_quat = torch.tensor(
        [
            [0.9698, 0.2121, 0.1203, -0.0039],
            [0.7020, 0.1578, 0.4525, 0.5268],
            [0.6766, 0.3176, 0.5179, 0.4161],
            [0.9085, 0.4020, 0.1139, -0.0025],
            [0.9698, 0.2121, 0.1203, -0.0039],
        ]
    )
    n_inter = args.num_views // 5
    args.num_views = n_inter * 5
    t_out = np.linspace(t_in[0], t_in[-1], n_inter * int(t_in[-1])).astype(np.float32)
    scales = np.array([2.0, 2.0, 2.0, 2.0, 2.0]).astype(np.float32)

    s_new = CubicSpline(t_in, scales, bc_type="periodic")
    s_new = s_new(t_out)

    q_new = CubicSpline(t_in, pose_quat.detach().cpu().numpy(), bc_type="periodic")
    q_new = q_new(t_out)
    q_new = q_new / np.linalg.norm(q_new, 2, 1)[:, None]
    q_new = torch.from_numpy(q_new).float()

    render_poses = []
    for i, (new_q, scale) in enumerate(zip(q_new, s_new)):
        new_q = new_q.unsqueeze(0)
        R = util.quat_to_rot(new_q)
        t = R[:, :, 2] * scale
        new_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        new_pose[:, :3, :3] = R
        new_pose[:, :3, 3] = t
        render_poses.append(new_pose)
    render_poses = torch.cat(render_poses, dim=0)
else:
    render_poses = torch.from_numpy(render_poses[..., :3, :4])
    if render_poses.shape[0] > args.num_views:
        n_inter = render_poses.shape[0]//args.num_views
        render_poses = render_poses[::n_inter]
        

render_rays = util.gen_rays(
    render_poses,
    W,
    H,
    focal * args.scale,
    z_near,
    z_far,
    c=c * args.scale if c is not None else None,
).to(device=device)
# (NV, H, W, 8)

focal = focal.to(device=device)
if args.dataset_format == 'dvr_dtu':
    source = torch.tensor(list(map(int, args.source.split())), dtype=torch.long)
else:
    source = random.sample(list(i_train), int(args.nviews[0]))
NS = len(source)
random_source = NS == 1 and source[0] == -1

if renderer.n_coarse < 64:
    # Ensure decent sampling resolution
    renderer.n_coarse = 64
    renderer.n_fine = 128

with torch.no_grad():
    print("Encoding source view(s)")
    if random_source:
        src_view = torch.randint(0, NV, (1,))
    else:
        src_view = source

    net.encode(
        images[src_view].unsqueeze(0),
        poses[src_view].unsqueeze(0).to(device=device),
        focal,
        c=c,
    )

    print("Rendering", args.num_views * H * W, "rays")
    all_rgb_fine = []
    for rays in tqdm.tqdm(
        torch.split(render_rays.view(-1, 8), args.ray_batch_size, dim=0)
    ):
        rgb, _depth = render_par(rays[None])
        all_rgb_fine.append(rgb[0])
    _depth = None
    rgb_fine = torch.cat(all_rgb_fine)
    # rgb_fine (V*H*W, 3)

    frames = rgb_fine.view(-1, H, W, 3)

print("Writing video")
vid_name = "{:04}".format(args.subset)
if args.split == "test":
    vid_name = "t" + vid_name
elif args.split == "val":
    vid_name = "v" + vid_name
vid_name += "_v" + "_".join(map(lambda x: "{:03}".format(x), source))
vid_path = os.path.join(args.visual_path, args.name, nv, "video" + vid_name + ".mp4")
viewimg_path = os.path.join(
    args.visual_path, args.name, nv, "video" + vid_name + "_view.jpg"
)
imageio.mimwrite(
    vid_path, (frames.cpu().numpy() * 255).astype(np.uint8), fps=args.fps, quality=8
)

img_np = (ori_images[src_view].permute(0, 2, 3, 1) * 0.5 + 0.5).numpy()
img_np = (img_np * 255).astype(np.uint8)
img_np = np.hstack((*img_np,))
imageio.imwrite(viewimg_path, img_np)

print("Wrote to", vid_path, "view:", viewimg_path)
