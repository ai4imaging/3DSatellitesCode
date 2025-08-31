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

import os
import random
import time
import torch
import numpy as np
from PIL import Image
from random import randint
from scene.cameras import Camera
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import getRt, get_c2w
from scipy.spatial.transform import Rotation as R, Slerp
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # rate = 0.1
    rate = 0.005
    css_center = gaussians.get_xyz.data.mean(0).detach().cpu().numpy()
    css_radius = np.sqrt(np.sum((gaussians.get_xyz.data.detach().cpu().numpy() - css_center) ** 2, axis=1)).max() * 1.2
    print('css_center: ', css_center, css_radius)
    start_time = time.time()
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_stack = random.sample(viewpoint_stack, 4)
            viewpoint_stack = viewpoint_stack + viewpoint_stack + viewpoint_stack
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()


        if iteration % 500 == 0 and iteration>=2000 and iteration < 8000:

            cam_trains = scene.getTrainCameras().copy()

            for cam_id, cam_ in enumerate(cam_trains):
                with torch.no_grad():
                    render_pkg = render(cam_, gaussians, pipe, bg)
                    im = render_pkg["render"]
                    gt_image = cam_.original_image.cuda()
                    L1_ = (1.0 - opt.lambda_dssim) * l1_loss(im, gt_image) + opt.lambda_dssim * (
                                1.0 - ssim(im, gt_image))
                    ls_ = L1_
                    # print('Loss:',cam_.image_name, L1_.item())

                    c2w_ = get_c2w(cam_.R, cam_.T)
                    c2ws = generate_complete_c2w_matrices(c2w_, rate, rate, rate, 200)
                    cam_back = None

                    for i, c2w in enumerate(c2ws):
                        R, T = getRt(c2w)
                        cam = Camera(0, R, T, cam_.FoVx, cam_.FoVx, cam_.original_image,
                                     None, cam_.image_name, cam_.uid)
                        render_pkg = render(cam, gaussians, pipe, bg)
                        image = render_pkg["render"]

                        L1 = (1.0 - opt.lambda_dssim) * l1_loss(image, gt_image) + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                        ls = L1

                        if ls < ls_:
                            ls_ = ls
                            cam_back = cam

                    if cam_back is not None:
                        scene.replaceCameras(cam_, cam_back)
                        print('Replaced',cam_.image_name, ls)
        if iteration % 3000 == 0:
            rate = rate/2
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 2 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            #
            if iteration >= 1000 and iteration < 12000 and (iteration + 1) % 5000 == 0:
                gaussians.refine()
            #
            if iteration == 29500:
                gaussians.refine()
                scene.save(iteration + 1)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                os.makedirs('log', exist_ok=True)
                for cam in scene.getTrainCameras():
                    render_pkg = render(cam, gaussians, pipe, bg)
                    image = render_pkg["render"]
                    image = np.clip(image.detach().cpu().numpy(), 0, 1)
                    image_array = (image * 255).astype(np.uint8)
                    image_array = np.transpose(image_array, (1, 2, 0))
                    im = Image.fromarray(image_array)
                    im.save('log/raw' +str(iteration)+"_"+ str(cam.image_name)  + ".png")
                scene.save(iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    end_time = time.time()
    print('gaussian number:', gaussians.get_xyz.shape[0])
    print('Time:', (end_time - start_time)/60, 'min')


def generate_complete_c2w_matrices(base_c2w, vertical_dist_coeff, parallel_dist_coeff, orientation_noise,num_matrices):
    def move_camera(base_c2w, vertical_dist_coeff, parallel_dist_coeff):
        rotation_matrix = base_c2w[:3, :3]
        forward_vector = -rotation_matrix[:, 2]
        right_vector = rotation_matrix[:, 0]
        up_vector = rotation_matrix[:, 1]

        # 计算垂直和平行方向的移动
        vertical_movement = forward_vector * vertical_dist_coeff * np.random.uniform(-1, 1)
        parallel_movement = (right_vector * np.random.uniform(-1, 1) +
                             up_vector * np.random.uniform(-1, 1)) * parallel_dist_coeff

        # 计算新的位置
        new_position = base_c2w[:3, 3] + vertical_movement + parallel_movement

        # 构造新的 c2w 矩阵，保持旋转矩阵不变，仅更新位置
        new_c2w = np.zeros_like(base_c2w)
        new_c2w[:3, :3] = rotation_matrix
        new_c2w[:3, 3] = new_position
        new_c2w[3, 3] = 1

        return new_c2w

    def add_orientation_noise(rotation_matrix, orientation_noise):
        if orientation_noise > 0:
            # 随机生成旋转轴和旋转角度
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(-orientation_noise, orientation_noise)
            # 生成旋转并应用到原始旋转矩阵
            noise_rotation = R.from_rotvec(axis * angle).as_matrix()
            return rotation_matrix @ noise_rotation
        return rotation_matrix

    # 根据位置变化和朝向随机生成一组 c2w 矩阵
    complete_c2w_matrices = []
    for _ in range(num_matrices):
        # 基于位置变化生成新的 c2w 矩阵
        c2w = move_camera(base_c2w, vertical_dist_coeff, parallel_dist_coeff)

        # 在原始旋转矩阵上施加朝向随机
        c2w[:3, :3] = add_orientation_noise(c2w[:3, :3], orientation_noise)

        complete_c2w_matrices.append(c2w)

    return complete_c2w_matrices

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[i*5000 for i in range(1, 7)]+[1, 1000, 2000, 3000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10000,20000,30000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    cams = training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, cams)

    # All done
    print("\nTraining complete.")
