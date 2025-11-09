# tools/generate_images_for_clip.py
import json
import cv2
import torch
import traceback
import numpy as np
import os
import os.path as osp
import shutil
import argparse
from PIL import Image
from run_infinity import *

torch._dynamo.config.cache_size_limit = 64

def process_short_text(short_text):
    if '--' in short_text:
        processed_text = short_text.split('--')[0]
        if processed_text:
            short_text = processed_text
    return short_text

def get_prompt_id(prompt):
    """生成prompt的唯一ID"""
    import hashlib
    return hashlib.md5(prompt.encode()).hexdigest()[:8]

def main():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    
    # 生成相关参数
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--coco30k_prompts', type=int, default=0, choices=[0,1])
    parser.add_argument('--jsonl_filepath', type=str, default='')
    parser.add_argument('--long_caption_fid', type=int, default=1, choices=[0,1])
    parser.add_argument('--total_prompts', type=int, default=-1)
    parser.add_argument('--n_samples', type=int, default=1, help='每个prompt生成的样本数')
    
    # 分布式参数
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    
    args = parser.parse_args()
    
    # 处理cfg参数
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    
    # 设置随机种子
    np.random.seed(0)
    torch.manual_seed(0)
    
    # 准备prompt列表
    lines4infer = []
    
    # 处理jsonl文件（COCO格式）
    if args.jsonl_filepath and osp.exists(args.jsonl_filepath):
        print(f"Loading prompts from {args.jsonl_filepath}")
        with open(args.jsonl_filepath, 'r') as f:
            for line in f:
                meta = json.loads(line)
                if args.long_caption_fid:
                    prompt = meta.get('long_caption', '')
                else:
                    prompt = meta.get('text', '')
                if not prompt:
                    continue
                lines4infer.append({
                    'prompt': prompt,
                    'h_div_w': meta.get('h_div_w', 1.0),
                    'infer_type': 'val/laion_coco_long_caption',
                    'original_meta': meta
                })
    else:
        # 默认prompt列表
        prompt_list = [
            'A high-contrast photo of a panda riding a horse.',
            'A beautiful sunset over the ocean.',
            'A cat sitting on a windowsill.',
        ]
        for prompt in prompt_list:
            lines4infer.append({
                'prompt': prompt,
                'h_div_w': 1.0,
                'infer_type': 'infer/free_prompt',
            })
    
    # 限制处理的prompt数量
    if args.total_prompts > 0:
        lines4infer = lines4infer[:args.total_prompts]
    
    # 分配prompt到不同GPU
    total_prompts = len(lines4infer)
    prompts_per_gpu = total_prompts // args.world_size
    start_idx = args.rank * prompts_per_gpu
    end_idx = start_idx + prompts_per_gpu if args.rank < args.world_size - 1 else total_prompts
    
    local_lines = lines4infer[start_idx:end_idx]
    
    print(f'GPU {args.rank}: Processing {len(local_lines)} prompts (total: {total_prompts})')
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    images_dir = osp.join(args.out_dir, 'generated_images')
    os.makedirs(images_dir, exist_ok=True)
    
    # 加载模型
    from run_infinity import load_tokenizer, load_visual_tokenizer, load_transformer
    from run_infinity import dynamic_resolution_h_w, h_div_w_templates
    
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)
    
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    infinity = infinity.to(device)
    
    # 存储元数据
    metadata = []
    
    for i, infer_data in enumerate(local_lines):
        try:
            prompt = infer_data['prompt']
            processed_prompt = process_short_text(prompt)
            prompt_id = get_prompt_id(processed_prompt)
            
            print(f'GPU {args.rank}: Processing prompt {i+1}/{len(local_lines)}: {prompt_id}')
            
            # 设置分辨率
            h_div_w = infer_data.get('h_div_w', 1.0)
            h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
            scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
            scale_schedule = [(1, h, w) for (t, h, w) in scale_schedule]
            
            if args.apply_spatial_patchify:
                vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
            else:
                vae_scale_schedule = scale_schedule
                
            tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']
            
            # 为每个prompt生成多个样本
            for sample_idx in range(args.n_samples):
                # 生成图像（使用不同的随机种子）
                seed = args.rank * 10000 + i * 100 + sample_idx
                
                # 调用infinity生成函数
                generated_img = gen_one_img(
                    infinity, vae, text_tokenizer, text_encoder, processed_prompt,
                    g_seed=seed, gt_leak=0, gt_ls_Bl=[], tau_list=args.tau, 
                    cfg_sc=3, cfg_list=args.cfg, scale_schedule=scale_schedule,
                    cfg_insertion_layer=[args.cfg_insertion_layer], 
                    vae_type=args.vae_type, sampling_per_bits=args.sampling_per_bits
                )
                
                # 保存图像
                image_filename = f'gpu{args.rank}_{prompt_id}_sample{sample_idx}.jpg'
                image_path = osp.join(images_dir, image_filename)
                
                if isinstance(generated_img, Image.Image):
                    generated_img.save(image_path)
                elif hasattr(generated_img, 'cpu'):
                    cv2.imwrite(image_path, generated_img.cpu().numpy())
                else:
                    cv2.imwrite(image_path, generated_img)
                
                # 记录元数据
                metadata.append({
                    'image_path': image_path,
                    'prompt': processed_prompt,
                    'prompt_id': prompt_id,
                    'sample_idx': sample_idx,
                    'gpu_rank': args.rank,
                    'prompt_index': i,
                    'seed': seed,
                    'h_div_w': h_div_w,
                    'infer_type': infer_data.get('infer_type', 'unknown')
                })
                
                print(f'GPU {args.rank}: Saved sample {sample_idx+1}/{args.n_samples} to {image_path}')
            
            # 定期保存进度
            if i % 10 == 0:
                metadata_file = osp.join(args.out_dir, f'generation_metadata_gpu{args.rank}.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f'GPU {args.rank}: Progress {i+1}/{len(local_lines)} - Generated {len(metadata)} images')
                
        except Exception as e:
            print(f"GPU {args.rank}: Error processing prompt {i}: {e}")
            traceback.print_exc()
            continue
    
    # 最终保存元数据
    metadata_file = osp.join(args.out_dir, f'generation_metadata_gpu{args.rank}.json')
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f'GPU {args.rank}: Successfully saved metadata for {len(metadata)} images')
    except Exception as e:
        print(f'GPU {args.rank}: Error saving metadata: {e}')
    
    # 完成标记
    completion_file = osp.join(args.out_dir, f'generation_gpu{args.rank}.done')
    with open(completion_file, 'w') as f:
        f.write(f'Completed at {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Generated {len(metadata)} images from {len(local_lines)} prompts\n')
    
    print(f'GPU {args.rank}: Image generation completed successfully')

if __name__ == '__main__':
    main()