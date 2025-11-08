# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"   # <<< 目标卡，务必在 import torch 前设置
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import os.path as osp
import random
import cv2
import numpy as np
import torch

# 你的工具函数（已包含 gen_one_img_with_power / 加载器等）
from tools.run_infinity import (
    load_tokenizer,
    load_visual_tokenizer,
    load_transformer,
    gen_one_img_with_power,      # 你刚补充的函数
)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

# cudnn 优化（可选）
torch.backends.cudnn.benchmark = True
torch.cuda.set_device(0)         # 注意：在 CUDA_VISIBLE_DEVICES 映射后，这里用 0

def add_common_arguments(parser):
    parser.add_argument('--cfg', type=str, default='3')
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--pn', type=str, required=True, choices=['0.06M', '0.25M', '1M'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cfg_insertion_layer', type=int, default=0)
    parser.add_argument('--vae_type', type=int, required=True)
    parser.add_argument('--vae_path', type=str, required=True)
    parser.add_argument('--add_lvl_embeding_only_first_block', type=int, default=1, choices=[0,1])
    parser.add_argument('--use_bit_label', type=int, default=1, choices=[0,1])
    parser.add_argument('--model_type', type=str, default='infinity_2b')
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1, choices=[0,1])
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2, choices=[0,1,2])
    parser.add_argument('--use_scale_schedule_embedding', type=int, default=0, choices=[0,1])
    parser.add_argument('--sampling_per_bits', type=int, default=1, choices=[1,2,4,8,16])
    parser.add_argument('--text_encoder_ckpt', type=str, required=True)
    parser.add_argument('--text_channels', type=int, default=2048)
    parser.add_argument('--apply_spatial_patchify', type=int, default=0, choices=[0,1])
    parser.add_argument('--h_div_w_template', type=float, default=1.000)
    parser.add_argument('--use_flex_attn', type=int, default=0, choices=[0,1])
    parser.add_argument('--enable_positive_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--cache_dir', type=str, default='/dev/shm')
    parser.add_argument('--enable_model_cache', action='store_true', default=True)
    parser.add_argument('--checkpoint_type', type=str, default='torch', choices=['torch','torch_shard'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bf16', type=int, default=1, choices=[0,1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Infinity power-only inference")
    add_common_arguments(parser)

    # 功耗专用参数
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--power_csv', type=str, required=True,
                        help='功耗结果 CSV 路径（必填）')
    parser.add_argument('--power_hz', type=float, default=200.0,
                        help='NVML 采样频率(Hz)，建议 100~500')
    parser.add_argument('--nvml_index', type=int, default=0,
                        help='NVML 的设备索引；若已用 CUDA_VISIBLE_DEVICES 指定单卡，这里通常为 0')

    args = parser.parse_args()

    # 解析 cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    # 固定随机种子（可选）
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 载入组件
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)

    # 组装 scale_schedule
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - args.h_div_w_template))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]  # (t,h,w)

    # 构造输出目录
    os.makedirs(osp.dirname(osp.abspath(args.power_csv)), exist_ok=True)

    # 只测功耗：不做 warmup、不做时间统计、不保存图片
    with torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.bf16 else torch.float32):
        with torch.no_grad():
            gen_one_img_with_power(
                infinity_test=infinity,
                vae=vae,
                text_tokenizer=text_tokenizer,
                text_encoder=text_encoder,
                prompt=args.prompt,
                csv_path=args.power_csv,
                g_seed=args.seed,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=args.cfg,
                tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
                power_hz=args.power_hz,
                device_index=args.nvml_index,   # NVML 的设备索引
            )

    print(f"[DONE] power CSV saved to: {osp.abspath(args.power_csv)}")
