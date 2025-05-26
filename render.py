#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import cv2
import json

import imageio.v2
import torch
from torch.autograd import Variable
import lpips
import torch
from train import generate_complete_c2w_matrices
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import getRt, get_c2w
import numpy as np
from scene.cameras import Camera
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from datetime import datetime
import matplotlib.pyplot as plt
lpips_loss = lpips.LPIPS(net="alex").cuda()


def normalize_and_color_map(depth_tensor, path):
    # 确保输入是浮点类型
    depth_tensor = depth_tensor.float()
    depth_tensor = depth_tensor.cpu().numpy()
    depth_tensor[depth_tensor!=0] -= depth_tensor[depth_tensor!=0].min() + 0.052
    depth_tensor[depth_tensor < 0] = 0
    depth_tensor = depth_tensor/depth_tensor.max()*255
    depth_tensor = depth_tensor.astype(np.uint8)

    # colored_depth = cv2.applyColorMap(depth_tensor, cv2.COLORMAP_JET)

    # 反转颜色映射，使远处为蓝色，近处为红色
    # colored_depth = 255 - colored_depth

    # 使用imageio保存图像
    imageio.imwrite(path, depth_tensor)

    return

def process_depth(tensor):
    tensor[tensor == 0] = tensor[tensor != 0].min()
    return tensor


def normalize_tensor_and_plot_histogram(tensor):
    # 确保输入是浮点类型
    tensor = tensor.float()
    tensor = process_depth(tensor)
    # 归一化到0-1
    tensor_normalized = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    # 转换为0-255范围的uint8类型
    tensor_scaled = (tensor_normalized * 255).round().byte()

    # 计算0-255每个值的频率
    values, counts = torch.unique(tensor_scaled, return_counts=True)

    # 准备绘制的完整数据（包括未出现的值）
    full_counts = np.zeros(256, dtype=int)
    full_counts[values.cpu().numpy()] = counts.cpu().numpy()

    # 创建柱状图
    plt.figure(figsize=(12, 6))
    plt.bar(range(256), full_counts, color='blue')

    # 添加标题和标签
    plt.title("Frequency Distribution of Normalized Values")
    plt.xlabel("Pixel Value (0-255)")
    plt.ylabel("Frequency")

    # 显示图表
    plt.show()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, scene = None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    rate = 0.05
    ssim_ = []
    psnr_ = []
    lpips_ = []
    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rend = render(view, gaussians, pipeline, background)

        base_c2w = get_c2w(view.R, view.T)
        c2ws = generate_complete_c2w_matrices(base_c2w, rate,rate,0, 5)
        c2ws = np.concatenate([c2ws, np.array([base_c2w])], axis=0)
        gt = view.original_image[0:3, :, :]
        mx_ = None
        ssmx = 0
        psnrmx = 0
        lpipsmx = 0
        centered_img = None
        depth_img = None
        for i, c2w in enumerate(c2ws):
            R, T = getRt(c2w)
            cam = Camera(0, R, T, view.FoVx, view.FoVx, view.original_image,
                         None, view.image_name, view.uid)
            rendering = render(cam, gaussians, pipeline, background)

            ss, ps, lps, _, centered, depth = find_optimal_placement_and_compute_metrics(rendering["render"].mean(0), gt.mean(0),rendering['depth'], 450, cam.original_image.shape[-1])
            if ss > ssmx:
                ssmx = ss
                psnrmx = ps
                lpipsmx = lps
                mx_ = _
                centered_img = centered
                depth_img = rendering['depth']
        ssim_.append(ssmx)
        psnr_.append(psnrmx)
        lpips_.append(lpipsmx)
        # print('SSIM:', ssmx, 'PSNR:', psnrmx, 'LPIPS:', lpipsmx)
        torchvision.utils.save_image(mx_, os.path.join(render_path, 'mix_{0:05d}'.format(idx) + "_.png"))
        torchvision.utils.save_image(centered_img, os.path.join(render_path, 'centered_{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendering['render'], os.path.join(render_path, 'render_{0:05d}'.format(idx) + ".png"))
    result = {
        'ssim': sum(ssim_) / len(ssim_),
        'psnr': sum(psnr_) / len(psnr_),
        'lpips': sum(lpips_) / len(lpips_)
    }
    result_path = os.path.join(model_path, name, "ours_{}".format(iteration), "result")
    os.makedirs(os.path.join(result_path, "renders"), exist_ok=True)
    print(result)
    with open(os.path.join(result_path, f"result_{now_str}.txt"), 'w') as f:
        f.write(str(result))
    np.save(os.path.join(result_path, f"ssim_{now_str}.npy"), np.array(ssim_))
    np.save(os.path.join(result_path, f"psnr_{now_str}.npy"), np.array(psnr_))
    np.save(os.path.join(result_path, f"lpips_{now_str}.npy"), np.array(lpips_))

    ssim_ = np.array(ssim_)
    psnr_ = np.array(psnr_)
    lpips_ = np.array(lpips_)
    print("ssim train:",ssim_[::2].mean())
    print("psnr train:",psnr_[::2].mean())
    print("lpips train:",lpips_[::2].mean())
    print("ssim val:",ssim_[1::2].mean())
    print("psnr val:",psnr_[1::2].mean())
    print("lpips val:",lpips_[1::2].mean())


def find_optimal_placement_and_compute_metrics(img1, img2, depth,n,img_width):

    # Check if n is greater than img_width
    if n <= img_width:
        raise ValueError("n should be greater than img_width.")

    # Expand img2 to n x n and center it
    expanded_img2 = torch.zeros((n, n), dtype=img2.dtype).cuda()
    start = (n - img_width) // 2
    expanded_img2[start:start + img_width, start:start + img_width] = img2

    # Convert tensors to numpy arrays for skimage metrics computation
    img1_np = img1.detach().cpu().numpy()

    # Initialize variables to keep track of the best placement
    optimal_l2_loss = float('inf')
    optimal_ssim = 0
    optimal_psnr = 0
    optimal_lpips = 0
    optimal_x = 0
    optimal_y = 0

    # Slide img1 over the expanded img2 to find the best placement
    for y in range(n - img_width + 1):
        for x in range(n - img_width + 1):

            # Compute the composite image for this placement
            composite_img = expanded_img2.clone()
            composite_img[y:y + img_width, x:x + img_width] = img1
            current_l2_loss = torch.mean((img1 - expanded_img2[y:y + img_width, x:x + img_width]) ** 2)

            # Update optimal values if this placement is better
            if current_l2_loss < optimal_l2_loss:
                optimal_l2_loss = current_l2_loss
                optimal_x = x
                optimal_y = y

                # Calculate SSIM, PSNR and LPIPS between the placed img1 and the corresponding area in expanded img2
                optimal_ssim = ssim(img1_np, expanded_img2[y:y + img_width, x:x + img_width].detach().cpu().numpy(),
                                    data_range=1)
                optimal_psnr = psnr(img1_np, expanded_img2[y:y + img_width, x:x + img_width].detach().cpu().numpy(),
                                    data_range=1)
                optimal_lpips = lpips_loss(img1, expanded_img2[y:y + img_width, x:x + img_width]).item()

                # print("Optimal x, y, L2 loss, SSIM, PSNR, LPIPS:", optimal_x, optimal_y, optimal_l2_loss, optimal_ssim, optimal_psnr, optimal_lpips)
    expanded_img1 = torch.zeros((n, n), dtype=img1.dtype).cuda()
    expanded_img1[optimal_y:optimal_y + img_width, optimal_x:optimal_x + img_width] = img1
    expanded_depth = torch.ones((n, n), dtype=depth.dtype).cuda() * depth.min()
    expanded_depth[optimal_y:optimal_y + img_width, optimal_x:optimal_x + img_width] = depth
    print("Optimal x, y, L2 loss, SSIM, PSNR, LPIPS:", optimal_x, optimal_y, optimal_l2_loss, optimal_ssim, optimal_psnr, optimal_lpips)
    return optimal_ssim, optimal_psnr, optimal_lpips, (expanded_img2[optimal_y:optimal_y + img_width, optimal_x:optimal_x + img_width] + img1)/2, expanded_img1[(n-img_width)//2:(n-img_width)//2 + img_width, (n-img_width)//2:(n-img_width)//2 + img_width], expanded_depth[(n-img_width)//2:(n-img_width)//2 + img_width, (n-img_width)//2:(n-img_width)//2 + img_width]


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, scene)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, scene)


def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", default=True, action="store_true")
    parser.add_argument("--skip_test", default=False, action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)


if __name__ == "__main__":
    main()
