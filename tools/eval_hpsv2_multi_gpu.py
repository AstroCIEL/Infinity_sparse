import os
import os.path as osp
import hashlib
import time
import argparse
import json
import shutil
import glob
import re
import sys

import cv2
import hpsv2
import torch
import numpy as np
from pytorch_lightning import seed_everything

from run_infinity import *

# set environment variables
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'
os.environ['HPS_ROOT']="/root/autodl-tmp/Infinity_sparse"

def extract_key_val(text):
    pattern = r'<(.+?):(.+?)>'
    matches = re.findall(pattern, text)
    key_val = {}
    for match in matches:
        key_val[match[0]] = match[1].lstrip()
    return key_val

def main():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--rewrite_prompt', type=int, default=0, choices=[0,1])
    
    # 添加分布式参数
    parser.add_argument('--rank', type=int, default=0, help='GPU rank (0, 1, 2, ...)')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of GPUs')
    parser.add_argument('--total_prompts', type=int, default=-1, help='Total number of prompts to process')
    
    args = parser.parse_args()

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    all_prompts = hpsv2.benchmark_prompts(['anime','photo'],download=False)
    
    # 设置随机种子，确保不同GPU生成不同的图片
    seed_everything(args.seed + args.rank)  # 添加rank偏移确保不同GPU生成不同图片

    # 展平所有prompts以便分配
    flat_prompts = []
    for style, prompts in all_prompts.items():
        for idx, prompt in enumerate(prompts):
            flat_prompts.append((style, idx, prompt))
    
    # 如果指定了total_prompts，只处理前total_prompts个
    if args.total_prompts > 0:
        flat_prompts = flat_prompts[:args.total_prompts]
    
    # 计算每个GPU需要处理的prompt范围
    total_prompts = len(flat_prompts)
    prompts_per_gpu = total_prompts // args.world_size
    start_idx = args.rank * prompts_per_gpu
    end_idx = start_idx + prompts_per_gpu if args.rank < args.world_size - 1 else total_prompts
    
    local_prompts = flat_prompts[start_idx:end_idx]
    
    print(f'Rank {args.rank}: Processing {len(local_prompts)} prompts (total: {total_prompts})')

    if args.model_type == 'sdxl':
        from diffusers import DiffusionPipeline
        base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to(f"cuda:{args.rank}")  # 指定GPU

        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(f"cuda:{args.rank}")
    elif args.model_type == 'sd3':
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        pipe = pipe.to(f"cuda:{args.rank}")
    elif args.model_type == 'pixart_sigma':
        from diffusers import PixArtSigmaPipeline
        pipe = PixArtSigmaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16
        ).to(f"cuda:{args.rank}")
    elif args.model_type == 'flux_1_dev':
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(f"cuda:{args.rank}")
    elif args.model_type == 'flux_1_dev_schnell':
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to(f"cuda:{args.rank}")
    elif 'infinity' in args.model_type:
        # load text encoder
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load vae
        vae = load_visual_tokenizer(args)
        # load infinity
        infinity = load_transformer(vae, args)
        # 将模型移动到指定GPU
        device = torch.device(f"cuda:0")
        vae = vae.to(device)
        infinity = infinity.to(device)

        if args.rewrite_prompt:
            from tools.prompt_rewriter import PromptRewriter
            prompt_rewriter = PromptRewriter(system='', few_shot_history=[])

    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)
    
    # 处理分配给当前GPU的prompt
    for i, (style, idx, prompt) in enumerate(local_prompts):
        try:
            image_save_file_path = os.path.join(args.outdir, style, f"{idx:05d}.jpg")
            os.makedirs(osp.dirname(image_save_file_path), exist_ok=True)
            
            # 检查文件是否已存在（避免重复生成）
            if osp.exists(image_save_file_path):
                print(f'Rank {args.rank}: Skipping existing {image_save_file_path}')
                continue

            tau = args.tau
            cfg = args.cfg
            if args.rewrite_prompt:
                refined_prompt = prompt_rewriter.rewrite(prompt)
                input_key_val = extract_key_val(refined_prompt)
                prompt = input_key_val['prompt']
                print(f'Rank {args.rank}: prompt: {prompt}, refined_prompt: {refined_prompt}')
            
            images = []
            for sample_idx in range(args.n_samples):
                t1 = time.time()
                if args.model_type == 'sdxl':
                    image = base(
                        prompt=prompt,
                        num_inference_steps=40,
                        denoising_end=0.8,
                        output_type="latent",
                    ).images
                    image = refiner(
                        prompt=prompt,
                        num_inference_steps=40,
                        denoising_start=0.8,
                        image=image,
                    ).images[0]
                elif args.model_type == 'sd3':
                    image = pipe(
                        prompt,
                        negative_prompt="",
                        num_inference_steps=28,
                        guidance_scale=7.0,
                        num_images_per_prompt=1,
                    ).images[0]
                elif args.model_type == 'flux_1_dev':
                    image = pipe(
                        prompt,
                        height=1024,
                        width=1024,
                        guidance_scale=3.5,
                        num_inference_steps=50,
                        max_sequence_length=512,
                        num_images_per_prompt=1,
                    ).images[0]
                elif args.model_type == 'flux_1_dev_schnell':
                    image = pipe(
                        prompt,
                        height=1024,
                        width=1024,
                        guidance_scale=0.0,
                        num_inference_steps=4,
                        max_sequence_length=256,
                        generator=torch.Generator("cpu").manual_seed(args.seed + args.rank + sample_idx)
                    ).images[0]
                elif args.model_type == 'pixart_sigma':
                    image = pipe(prompt).images[0]
                elif 'infinity' in args.model_type:
                    h_div_w_template = 1.000
                    scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
                    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
                    tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']
                    # 使用不同的种子确保不同GPU生成不同图片
                    g_seed = args.seed + args.rank * 1000 + i * 10 + sample_idx
                    image = gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, 
                                      g_seed=g_seed, tau_list=tau, cfg_sc=3, cfg_list=cfg, 
                                      scale_schedule=scale_schedule, cfg_insertion_layer=[args.cfg_insertion_layer], 
                                      vae_type=args.vae_type)
                else:
                    raise ValueError
                t2 = time.time()
                print(f'Rank {args.rank}: {args.model_type} infer one image takes {t2-t1:.2f}s')
                images.append(image)
            
            assert len(images) == 1
            for image in images:
                if 'infinity' in args.model_type:
                    cv2.imwrite(image_save_file_path, image.cpu().numpy())
                else:
                    image.save(image_save_file_path)
            
            # 定期打印进度
            if i % 10 == 0:
                print(f'Rank {args.rank}: Progress {i+1}/{len(local_prompts)}')
                
        except Exception as e:
            print(f"Rank {args.rank}: Error processing prompt {i} (style: {style}, idx: {idx}): {e}")
            import traceback
            traceback.print_exc()

    # 添加完成标记文件
    completion_file = osp.join(args.outdir, f'rank{args.rank}.done')
    with open(completion_file, 'w') as f:
        f.write(f'Completed at {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Processed {len(local_prompts)} prompts\n')

    print(f'Rank {args.rank}: Process completed successfully')

if __name__ == '__main__':
    main()