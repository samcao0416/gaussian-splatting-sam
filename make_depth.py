from scene import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from PIL import Image
from scene.cameras import Camera
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.image_utils import warpped_depth
from utils.general_utils import inverse_sigmoid
from tqdm import tqdm
import numpy as np
import os
import sys
import torch
from torch import nn

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Make depth map")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    gaussians = GaussianModel(1)

    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, "", args.eval)
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
    else:
        assert False, "Could not recognize scene type!"
    cameras_extent = scene_info.nerf_normalization["radius"]
    gaussians.create_from_pcd(scene_info.point_cloud, cameras_extent)
    opacities = inverse_sigmoid(0.9999 * torch.ones((gaussians._opacity.shape[0], 1), dtype=torch.float, device="cuda"))
    gaussians._opacity = nn.Parameter(opacities.requires_grad_(True))

    camera_list = []
    for _id, cam_info in enumerate(scene_info.train_cameras):
        orig_w, orig_h = cam_info.image.size
        gt_image = torch.zeros((3, orig_h, orig_w), dtype=torch.float32)
        loaded_mask = None
        camera = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, depth=None, depth_mask=None, uid=_id, data_device=args.data_device)
        camera.original_image = None
        camera.gt_alpha_mask = None
        camera_list.append(camera)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    save_dir = os.path.join(dataset.source_path, "depths")
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for camera in tqdm(camera_list):
            render_pkg = render(camera, gaussians, pipe, background)
            accum_alpha = 1.0 - render_pkg["final_T"]
            depth = torch.where(accum_alpha == 0.0, torch.zeros_like(render_pkg["depth"]), render_pkg["depth"] / accum_alpha)
            depth = warpped_depth(depth)
            depth = Image.fromarray(torch.stack([depth[0] * 255.0, accum_alpha[0] * 255.0], dim=2).detach().cpu().numpy().astype(np.uint8), mode="LA")
            # depth = Image.fromarray((render_pkg["depth"][0]).detach().cpu().numpy().astype(np.float32))
            # accum_alpha = Image.fromarray((1.0 - render_pkg["final_T"][0]).detach().cpu().numpy().astype(np.float32))
            depth.save(os.path.join(save_dir, camera.image_name + ".png"))
            # accum_alpha.save(os.path.join(save_dir, camera.image_name + "_accum_alpha" + ".tiff"))