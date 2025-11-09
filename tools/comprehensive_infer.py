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

def main():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--rewrite_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--coco30k_prompts', type=int, default=0, choices=[0,1])
    parser.add_argument('--save4fid_eval', type=int, default=0, choices=[0,1])
    parser.add_argument('--save_recons_img', type=int, default=0, choices=[0,1])
    parser.add_argument('--jsonl_filepath', type=str, default='')
    parser.add_argument('--long_caption_fid', type=int, default=1, choices=[0,1])
    parser.add_argument('--fid_max_examples', type=int, default=-1)
    parser.add_argument('--n_samples', type=int, default=1)
    
    # 添加分布式参数
    parser.add_argument('--rank', type=int, default=0, help='GPU rank (0, 1, 2, ...)')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of GPUs')
    parser.add_argument('--total_prompts', type=int, default=-1, help='Total number of prompts to process')
    
    args = parser.parse_args()
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    
    # 设置随机种子，确保不同GPU生成不同的图片
    np.random.seed(0)
    torch.manual_seed(0)
    
    # 准备prompt列表（与原始代码相同）
    prompt_list = [
        'A high-contrast photo of a panda riding a horse. The panda is wearing a wizard hat and is reading a book. The horse is standing on a street against a gray concrete wall. Colorful flowers and the word "PEACE" are painted on the wall. Green grass grows from cracks in the street. DSLR photograph. daytime lighting.',
        # ... 原有的prompt列表保持不变
    ]
    
    lines4infer = []
    for prompt in prompt_list:
        lines4infer.append({
            'prompt': prompt,
            'h_div_w': 1.0,
            'infer_type': 'infer/free_prompt',
        })
    
    # 处理COCO30k prompts（与原始代码相同）
    if args.coco30k_prompts:
        from T2IBenchmark.datasets import get_coco_30k_captions, get_coco_fid_stats
        id2caption = get_coco_30k_captions()
        captions = []
        ids = []
        for d in id2caption.items():
            ids.append(d[0])
            captions.append(d[1])
        np.random.shuffle(captions)
        lines4infer = [{'prompt': prompt, 'h_div_w': 1.0, 'infer_type': 'infer/coco30k_prompt'} for prompt in captions]
    
    # 处理jsonl文件（与原始代码相同）
    if args.jsonl_filepath:
        lines4infer = []
        with open(args.jsonl_filepath, 'r') as f:
            cnt = 0
            for line in f:
                meta = json.loads(line)
                gt_image_path = meta['image_path']
                assert osp.exists(gt_image_path), gt_image_path
                if args.long_caption_fid:
                    prompt = meta['long_caption']
                else:
                    prompt = meta['text']
                if not prompt:
                    continue
                lines4infer.append({
                    'prompt': prompt,
                    'h_div_w': meta['h_div_w'],
                    'infer_type': 'val/laion_coco_long_caption',
                    'gt_image_path': gt_image_path,
                    'meta_line': line,
                })
    
    if args.fid_max_examples > 0:
        lines4infer = lines4infer[:args.fid_max_examples]
    
    # 关键修改：将prompt分配给不同的GPU
    if args.total_prompts > 0:
        # 如果指定了total_prompts，只处理前total_prompts个
        lines4infer = lines4infer[:args.total_prompts]
    
    # 计算每个GPU需要处理的prompt范围
    total_prompts = len(lines4infer)
    prompts_per_gpu = total_prompts // args.world_size
    start_idx = args.rank * prompts_per_gpu
    end_idx = start_idx + prompts_per_gpu if args.rank < args.world_size - 1 else total_prompts
    
    local_lines = lines4infer[start_idx:end_idx]
    
    print(f'GPU {args.rank}: Processing {len(local_lines)} prompts (total: {total_prompts})')
    
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = osp.join('output', osp.basename(osp.dirname(args.model_path)), 
                          osp.splitext(osp.basename(args.model_path))[0], 
                          f'coco30k_infer' if args.coco30k_prompts else 'comprehensive_infer')
    
    print(f'GPU {args.rank}: Saving to {out_dir}')
    
    # 加载模型（每个GPU独立加载）
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 创建输出目录
    if osp.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        if args.rank == 0:  # 只在主GPU复制文件
            shutil.copyfile(__file__, osp.join(out_dir, osp.basename(__file__)))
    
    # 处理分配给当前GPU的prompt
    jsonl_list = []
    for i, infer_data in enumerate(local_lines):
        try:
            prompt = infer_data['prompt']
            prompt = process_short_text(prompt)
            prompt_id = get_prompt_id(prompt)
            
            # 添加GPU编号到文件名，避免冲突
            save_file = osp.join(out_dir, 'pred', f'gpu{args.rank}_{prompt_id}.jpg')
            
            if osp.exists(save_file):
                print(f'GPU {args.rank}: Skipping existing {save_file}')
                continue
            
            # 原有的生成逻辑保持不变
            h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - infer_data['h_div_w']))]
            scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
            scale_schedule = [(1, h, w) for (t, h, w) in scale_schedule]
            
            if args.apply_spatial_patchify:
                vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
            else:
                vae_scale_schedule = scale_schedule
                
            tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']

            # 处理GT图片（如果有）
            gt_ls_Bl = []
            if 'gt_image_path' in infer_data:
                gt_img, recons_img, all_bit_indices = joint_vi_vae_encode_decode(
                    vae, infer_data['gt_image_path'], vae_scale_schedule, device, tgt_h, tgt_w)
                gt_ls_Bl = all_bit_indices
            else:
                if args.save4fid_eval:
                    continue

            # 生成图片
            if args.coco30k_prompts or args.save4fid_eval:
                concate_img = gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, 
                                         g_seed=args.rank * 1000 + i,  # 使用不同的种子
                                         gt_leak=0, gt_ls_Bl=gt_ls_Bl, tau_list=args.tau, 
                                         cfg_sc=3, cfg_list=args.cfg, scale_schedule=scale_schedule, 
                                         cfg_insertion_layer=[args.cfg_insertion_layer], 
                                         vae_type=args.vae_type, sampling_per_bits=args.sampling_per_bits)
            else:
                # 多样本生成逻辑
                img_list = []
                g_seed = args.rank * 1000 + i if args.n_samples == 1 else None
                
                # 原有的多样本生成逻辑
                tmp_img_list = []
                for j in range(args.n_samples):
                    seed = g_seed + j * 100 if g_seed is not None else None
                    tmp_img_list.append(gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, 
                                                   g_seed=seed, gt_leak=0, gt_ls_Bl=gt_ls_Bl, 
                                                   t5_path=None, tau_list=args.tau, cfg_sc=3, 
                                                   cfg_list=args.cfg, scale_schedule=scale_schedule, 
                                                   cfg_insertion_layer=[args.cfg_insertion_layer], 
                                                   vae_type=args.vae_type, sampling_per_bits=1, top_k=0))
                img_list.append(np.concatenate(tmp_img_list, axis=1))
                
                # ... 其他采样参数的生成逻辑

            # 保存图片
            os.makedirs(osp.dirname(save_file), exist_ok=True)
            if hasattr(concate_img, 'cpu'):  # 如果是tensor
                cv2.imwrite(save_file, concate_img.cpu().numpy())
            else:  # 如果是PIL Image或numpy数组
                if isinstance(concate_img, Image.Image):
                    concate_img.save(save_file)
                else:
                    cv2.imwrite(save_file, concate_img)
            
            infer_data['image_path'] = osp.abspath(save_file)
            
            # 保存GT图片（如果需要）
            if args.save4fid_eval and 'gt_image_path' in infer_data:
                save_file_gt = osp.join(out_dir, 'gt', f'gpu{args.rank}_{prompt_id}.jpg')
                os.makedirs(osp.dirname(save_file_gt), exist_ok=True)
                if isinstance(gt_img, Image.Image):
                    gt_img.save(save_file_gt)
                else:
                    cv2.imwrite(save_file_gt, gt_img)
                
                if args.save_recons_img:
                    save_file_recons = osp.join(out_dir, 'recons', f'gpu{args.rank}_{prompt_id}.jpg')
                    os.makedirs(osp.dirname(save_file_recons), exist_ok=True)
                    if isinstance(recons_img, Image.Image):
                        recons_img.save(save_file_recons)
                    else:
                        cv2.imwrite(save_file_recons, recons_img)

            # 更新元数据
            jsonl_list.append(json.dumps(infer_data) + '\n')
            
            # 定期保存进度
            if i % 10 == 0:
                jsonl_file = osp.join(out_dir, f'meta_info_gpu{args.rank}.jsonl')
                with open(jsonl_file, 'w') as f:
                    f.writelines(jsonl_list)
                print(f'GPU {args.rank}: Progress {i+1}/{len(local_lines)}')
                
        except Exception as e:
            print(f"GPU {args.rank}: Error processing prompt {i}: {e}")
            traceback.print_exc()
    
    # 最终保存元数据
    jsonl_file = osp.join(out_dir, f'meta_info_gpu{args.rank}.jsonl')
    try:
        with open(jsonl_file, 'w') as f:
            f.writelines(jsonl_list)
        print(f'GPU {args.rank}: Successfully saved {len(jsonl_list)} items to {jsonl_file}')
    except Exception as e:
        print(f'GPU {args.rank}: Error saving metadata: {e}')

    # 添加一个完成标记文件
    completion_file = osp.join(out_dir, f'gpu{args.rank}.done')
    with open(completion_file, 'w') as f:
        f.write(f'Completed at {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Processed {len(jsonl_list)} prompts\n')

    print(f'GPU {args.rank}: Process completed successfully')

if __name__ == '__main__':
    main()