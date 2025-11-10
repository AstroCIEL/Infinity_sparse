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
import traceback

import cv2
import tqdm
import torch
import numpy as np
from pytorch_lightning import seed_everything

from run_infinity import *

# set environment variables
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'

def main():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--outdir', type=str, default='/DISK1/home/yx_zhao31/Infinity/output')
    parser.add_argument('--n_samples', type=int, default=5)
    parser.add_argument('--metadata_file', type=str, default='evaluation/gen_eval/prompts/evaluation_metadata.jsonl')
    parser.add_argument('--rewrite_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--load_rewrite_prompt_cache', type=int, default=1, choices=[0,1])
    
    # 添加分布式参数
    parser.add_argument('--rank', type=int, default=0, help='GPU rank (0, 1, 2, ...)')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of GPUs')
    parser.add_argument('--total_prompts', type=int, default=-1, help='Total number of prompts to process')
    
    args = parser.parse_args()

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    
    # 加载metadata文件
    with open(args.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    
    # 限制处理的prompt数量
    if args.total_prompts > 0:
        metadatas = metadatas[:args.total_prompts]
    
    # 计算每个GPU需要处理的prompt范围
    total_prompts = len(metadatas)
    prompts_per_gpu = total_prompts // args.world_size
    start_idx = args.rank * prompts_per_gpu
    end_idx = start_idx + prompts_per_gpu if args.rank < args.world_size - 1 else total_prompts
    
    local_metadatas = metadatas[start_idx:end_idx]
    
    print(f'GPU {args.rank}: Processing {len(local_metadatas)} prompts (total: {total_prompts})')
    
    prompt_rewrite_cache_file = osp.join('evaluation/gen_eval', 'prompt_rewrite_cache.json')
    if osp.exists(prompt_rewrite_cache_file):
        with open(prompt_rewrite_cache_file, 'r') as f:
            prompt_rewrite_cache = json.load(f)
    else:
        prompt_rewrite_cache = {}

    # 加载模型（每个GPU独立加载）
    if 'infinity' in args.model_type:
        # load text encoder
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load vae
        vae = load_visual_tokenizer(args)
        # load infinity
        infinity = load_transformer(vae, args)


        if args.rewrite_prompt:
            from tools.prompt_rewriter import PromptRewriter
            prompt_rewriter = PromptRewriter(system='', few_shot_history=[])

    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)
    
    # 处理分配给当前GPU的metadata
    for index, metadata in enumerate(local_metadatas):
        try:
            # 计算全局索引
            global_index = start_idx + index
            outpath = os.path.join(args.outdir, f"{global_index:0>5}")
            os.makedirs(outpath, exist_ok=True)
            
            prompt = metadata['prompt']
            print(f"GPU {args.rank}: Prompt ({global_index: >3}/{total_prompts}): '{prompt}'")

            sample_path = os.path.join(outpath, "samples")
            os.makedirs(sample_path, exist_ok=True)
            
            # 保存metadata
            with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
                json.dump(metadata, fp)

            tau = args.tau
            cfg = args.cfg
            
            # prompt重写逻辑
            if args.rewrite_prompt:
                old_prompt = prompt
                if args.load_rewrite_prompt_cache and prompt in prompt_rewrite_cache:
                    prompt = prompt_rewrite_cache[prompt]
                else:
                    refined_prompt = prompt_rewriter.rewrite(prompt)
                    input_key_val = extract_key_val(refined_prompt)
                    prompt = input_key_val['prompt']
                    prompt_rewrite_cache[prompt] = prompt
                print(f'GPU {args.rank}: old_prompt: {old_prompt}, refined_prompt: {prompt}')
            
            images = []
            for sample_j in range(args.n_samples):
                print(f"GPU {args.rank}: Generating {sample_j+1} of {args.n_samples}, prompt={prompt}")
                t1 = time.time()
                
                if 'infinity' in args.model_type:
                    h_div_w_template = 1.000
                    scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
                    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
                    tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']
                    
                    # 设置不同的种子确保不同GPU生成不同的图片
                    seed = args.seed + args.rank * 1000 + index * 10 + sample_j
                    seed_everything(seed)
                    
                    image = gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, 
                                     tau_list=tau, cfg_sc=3, cfg_list=cfg, 
                                     scale_schedule=scale_schedule, 
                                     cfg_insertion_layer=[args.cfg_insertion_layer], 
                                     vae_type=args.vae_type)
                else:
                    raise ValueError(f"Unsupported model type: {args.model_type}")
                
                t2 = time.time()
                print(f'GPU {args.rank}: {args.model_type} infer one image takes {t2-t1:.2f}s')
                images.append(image)
            
            # 保存图片
            for i, image in enumerate(images):
                save_file = os.path.join(sample_path, f"{i:05}.jpg")
                if 'infinity' in args.model_type:
                    if hasattr(image, 'cpu'):  # 如果是tensor
                        cv2.imwrite(save_file, image.cpu().numpy())
                    else:
                        cv2.imwrite(save_file, image)
                else:
                    image.save(save_file)
            
            # 定期保存进度
            if (index + 1) % 10 == 0:
                print(f'GPU {args.rank}: Progress {index+1}/{len(local_metadatas)}')
                
        except Exception as e:
            print(f"GPU {args.rank}: Error processing prompt {index}: {e}")
            traceback.print_exc()
    
    # 保存prompt重写缓存（每个GPU都保存，但只有最后一个会生效）
    try:
        with open(prompt_rewrite_cache_file, 'w') as f:
            json.dump(prompt_rewrite_cache, f, indent=2)
        print(f'GPU {args.rank}: Successfully saved prompt rewrite cache')
    except Exception as e:
        print(f'GPU {args.rank}: Error saving prompt rewrite cache: {e}')

    # 添加完成标记文件
    completion_file = osp.join(args.outdir, f'gpu{args.rank}.done')
    with open(completion_file, 'w') as f:
        f.write(f'Completed at {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Processed {len(local_metadatas)} prompts\n')

    print(f'GPU {args.rank}: Process completed successfully')

if __name__ == '__main__':
    main()