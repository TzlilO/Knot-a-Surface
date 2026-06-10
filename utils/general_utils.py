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
from math import cos, pi

import torch
import sys
from datetime import datetime
import numpy as np
import random
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution=None):
    if resolution is None:
        resized_image_PIL = pil_image
    else:
        resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_cosine_warm_restarts_lr_func(
    lr_init, lr_final, T_0, T_mult=1, N_cycles_before_decay=5, decay_factor=0.95,
    lr_delay_steps=0, lr_delay_mult=1.0
):
    """
    Cosine annealing with warm restarts and post-N cycle decay.
    Adapted from PyTorch's CosineAnnealingWarmRestarts with added decay envelope.
    After N_cycles_before_decay, scales the peak LR (effective_lr_init) by decay_factor
    for each subsequent cycle, enabling overall decay while preserving periodicity.
    No max_steps; runs indefinitely based on T_0 and T_mult.
    If lr_delay_steps>0, applies initial reverse cosine scaling as in original.
    :param N_cycles_before_decay: int, number of full cycles before starting decay.
    :param decay_factor: float (<1), multiplier for peak LR after N cycles.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        # Find current cycle and T_cur (iterative loop for cycle determination)
        current_step = step
        current_T = T_0
        cycle = 0
        while current_step >= current_T:
            current_step -= current_T
            current_T *= T_mult
            cycle += 1
        T_cur = current_step
        T_i = current_T

        # Apply decay to effective_lr_init if after N cycles
        effective_lr_init = lr_init
        if cycle > N_cycles_before_decay:
            effective_lr_init *= (decay_factor ** (cycle - N_cycles_before_decay))

        # Standard cosine within cycle, using effective_lr_init
        cosine_decay = lr_final + 0.5 * (effective_lr_init - lr_final) * (1 + np.cos(np.pi * T_cur / T_i))
        return delay_rate * cosine_decay
    return helper

def get_cosine_warm_restarts_lr_func2(
    lr_init, lr_final, T_0, T_mult=1, lr_delay_steps=0, lr_delay_mult=1.0
):
    """
    Cosine annealing with warm restarts learning rate decay function.
    Adapted from PyTorch's CosineAnnealingWarmRestarts.
    Implements periodic cosine decays with restarts, where cycle lengths grow by T_mult.
    No max_steps; runs indefinitely based on T_0 and T_mult.
    If lr_delay_steps>0, applies initial reverse cosine scaling as in original.
    :param T_0: int, initial cycle length in steps.
    :param T_mult: float, multiplier for subsequent cycle lengths (>=1).
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        # Find current cycle and T_cur (iterative but efficient for small num_cycles)
        current_step = step
        current_T = T_0
        while current_step >= current_T:
            current_step -= current_T
            current_T *= T_mult
        T_cur = current_step
        T_i = current_T

        cosine_decay = lr_final + 0.5 * (lr_init - lr_final) * (1 + np.cos(np.pi * T_cur / T_i))
        return delay_rate * cosine_decay
    return helper


def get_nurbs_lr_func(
        lr_init: float,
        lr_final: float,
        lr_mid: float,  # NEW: intermediate LR for refinement phase
        coarse_steps: int,  # Steps for coarse fitting
        refinement_steps: int,  # Steps for detail refinement
        max_steps: int
):
    """
    Two-phase LR schedule for NURBS optimization.

    Phase 1 (Coarse): Moderate LR with slow decay
        - Establish global surface shape
        - Allow large control point movements

    Phase 2 (Refinement): Lower LR with faster decay
        - After densification stabilizes
        - Fine-tune local details
    """

    def lr_func(step):
        if step < coarse_steps:
            # Phase 1: Slower decay, higher floor
            t = step / coarse_steps
            # Cosine annealing (smoother than exponential for coarse phase)
            return lr_init * (1 + cos(pi * t)) / 2 * (1 - lr_mid / lr_init) + lr_mid
        else:
            # Phase 2: Exponential decay from lr_mid to lr_final
            t = (step - coarse_steps) / (max_steps - coarse_steps)
            return lr_mid * (lr_final / lr_mid) ** t

    return lr_func
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=30000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp
    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling(s):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]
    return L

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    print(torch.cuda.is_available())
    # torch.cuda.set_device(torch.device("cuda"))
class VideoLogger:
    def __init__(self, max_video_frames=256):
        self.reconstructed_frames = dict()
        self.surf_normal_frames = dict()
        self.rend_normal_frames = dict()
        self.max_frames = max_video_frames
        self.executor = ThreadPoolExecutor(max_workers=2)  # Adjust max_workers as needed

    def add_frames(self, uid, reconstructed, normal, surf):
        """Add frames to the video buffers"""
        try:
            self.reconstructed_frames[uid].append(reconstructed)
            if normal is not None:
                self.rend_normal_frames[uid].append(normal)
            if surf is not None:
                self.surf_normal_frames[uid].append(surf)

        except KeyError:
            self.reconstructed_frames[uid] = [reconstructed]
            self.rend_normal_frames[uid] = [normal]
            self.surf_normal_frames[uid] = [surf]

        # Keep only the most recent frames if we exceed max_frames
        if len(self.reconstructed_frames[uid]) == self.max_frames:
            self.log_videos(uid)
            self.reconstructed_frames[uid] = []
            self.rend_normal_frames[uid] = []
            self.surf_normal_frames[uid] = []

    def log_videos(self, uid, scale_factor=1, fps=36):
        if uid not in self.reconstructed_frames:
            return  # No frames to log

        # Prepare data for async processing
        reconstructed_sequence = np.copy(np.stack(self.reconstructed_frames[uid], axis=0))
        if len(self.rend_normal_frames) > 0:
            normal_sequence = np.copy(np.stack(self.rend_normal_frames[uid], axis=0))
        if len(self.surf_normal_frames) > 0:
            depth_sequence = np.copy(np.stack(self.surf_normal_frames[uid], axis=0))

        # Start async processing
        self.executor.submit(self._process_and_upload_videos, uid, reconstructed_sequence, normal_sequence, depth_sequence, scale_factor, fps)
        # self._process_and_upload_videos(uid, reconstructed_sequence, normal_sequence, depth_sequence, scale_factor, fps)
        # Clear the frame buffers immediately
        self.reconstructed_frames[uid] = []
        self.rend_normal_frames[uid] = []
        self.surf_normal_frames[uid] = []

    @staticmethod
    def _downsample_frames(frames, scale_factor):
        resized = []
        for f in frames:
            h, w = f.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            resized_frame = cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA)
            resized.append(resized_frame)
        return np.stack(resized, axis=0)

    def _process_and_upload_videos(self, uid, reconstructed_sequence, normal_sequence, surf_sequence, scale_factor, fps):
        # Ensure uint8
        if reconstructed_sequence.dtype != np.uint8:
            reconstructed_sequence = np.clip(reconstructed_sequence * 255, 0, 255).astype(np.uint8)
        if normal_sequence.dtype != np.uint8:
            normal_sequence = np.clip(normal_sequence * 255, 0, 255).astype(np.uint8)


        if surf_sequence.dtype != np.uint8:
            surf_sequence = np.clip(surf_sequence * 255, 0, 255).astype(np.uint8)

        # Downsample frames
        reconstructed_sequence = self._downsample_frames(reconstructed_sequence, scale_factor)

        wandb.log({
            f"View_{uid}/Reconstructed_Sequence": wandb.Video(
                reconstructed_sequence.transpose(0, 3, 1, 2), fps=fps, format="gif"
            )})
        if len(normal_sequence) > 0:
            normal_sequence = self._downsample_frames(normal_sequence, scale_factor)
            wandb.log({
                f"View_{uid}/Normal_Sequence": wandb.Video(
                    normal_sequence.transpose(0, 3, 1, 2), fps=fps, format="gif")})

        if len(surf_sequence) > 0:
            surf_sequence = self._downsample_frames(surf_sequence, scale_factor)
            wandb.log({
                f"View_{uid}/Surf_Normal_Sequence": wandb.Video(
                    surf_sequence.transpose(0, 3, 1, 2), fps=fps, format="gif")})


        # Log to wandb

def wandb_logger(renders:dict, run_dict:dict, video: VideoLogger=None):
    """
    Log images and metrics to Weights & Biases with side-by-side comparisons and video support.
    """

    import wandb

    # Get GPU memory usage
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2  # Convert to MB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2  # Convert to MB
    else:
        gpu_memory_allocated = 0
        gpu_memory_reserved = 0

        # Update the log dictionary to include all loss components and GPU usage


    log_dict = {}
    for k,v in run_dict.items():
        # try:
        log_dict[k] = v
        # except:
    log_dict["GPU Memory Allocated (MB)"] = gpu_memory_allocated
    log_dict["GPU Memory Reserved (MB)"] = gpu_memory_reserved

    wandb.log(log_dict, step=run_dict["iteration"])

    ### Rest is Commented out until efficient animated logger is implemented
    # import cv2
    # # Ensure proper normalization for images
    # def normalize_image(img, n=255):
    #     if img.dtype != np.uint8:
    #         img = np.clip(img * n, 0, n).astype(np.uint8)
    #     return img
    #
    # def normalize_depth(depth):
    #     depth = depth.squeeze()
    #     depth_min = np.min(depth)
    #     depth_max = np.max(depth)
    #     normalized_depth = (depth - depth_min) / (depth_max - depth_min)
    #     return (normalized_depth * 255).astype(np.uint8)
    #
    # def normalize_normals(normals):
    #     return ((normals + 1) * 127.5).astype(np.uint8)
    #

    # Process images
    # pred_img = normalize_image(predicted_image.detach().cpu().permute(1, 2, 0).numpy())
    # normal_viz = normalize_normals(normal_map.detach().cpu().permute(1, 2, 0).numpy())
    # surf_viz = normalize_normals(surf_norm.detach().cpu().permute(1, 2, 0).numpy())
    # Ensure contiguous and proper dtype
    # pred_img = np.ascontiguousarray(pred_img, dtype=np.uint8)
    # surf_viz = np.ascontiguousarray(surf_viz, dtype=np.uint8)
    # normal_viz = np.ascontiguousarray(normal_viz, dtype=np.uint8)
    # Choose positions and font scale as desired
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # text_color = (255, 255, 255)  # White text
    # thickness = 2
    # line_type = cv2.LINE_AA

    # # Put text onto the predicted image
    # cv2.putText(pred_img, f"Iteration: {iteration}", (10, 30), font, 1.0, text_color, thickness, line_type)
    # cv2.putText(pred_img, f"PSNR: {psnr_score:.2f}", (10, 60), font, 1.0, text_color, thickness, line_type)
    # cv2.putText(pred_img, f"SSIM: {ssim_score:.2f}", (10, 90), font, 1.0, text_color, thickness, line_type)
    # cv2.putText(pred_img, f"Overall patches: {num_patches}", (10, 120), font, 1.0, text_color, thickness, line_type)
    #
    #
    # cv2.putText(surf_viz, f"Iteration: {iteration}", (10, 30), font, 1.0, text_color, thickness, line_type)
    # cv2.putText(surf_viz, f"PSNR: {psnr_score:.2f}", (10, 60), font, 1.0, text_color, thickness, line_type)
    # cv2.putText(surf_viz, f"SSIM: {ssim_score:.2f}", (10, 90), font, 1.0, text_color, thickness, line_type)
    # cv2.putText(surf_viz, f"Overall patches: {num_patches}", (10, 120), font, 1.0, text_color, thickness, line_type)
    #
    # cv2.putText(normal_viz, f"Iteration: {iteration}", (10, 30), font, 1.0, text_color, thickness, line_type)
    # cv2.putText(normal_viz, f"PSNR: {psnr_score:.2f}", (10, 60), font, 1.0, text_color, thickness, line_type)
    # cv2.putText(normal_viz, f"SSIM: {ssim_score:.2f}", (10, 90), font, 1.0, text_color, thickness, line_type)
    # cv2.putText(normal_viz, f"Overall patches: {num_patches}", (10, 120), font, 1.0, text_color, thickness, line_type)
    #

    # video.add_frames(uid, pred_img, None, None)

