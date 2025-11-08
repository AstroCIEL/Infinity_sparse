import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
from typing import List
import math
import time
import hashlib
import yaml
import argparse
import shutil
import re

import cv2
import numpy as np
import torch
torch._dynamo.config.cache_size_limit=64
import pandas as pd
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
from PIL import Image, ImageEnhance
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infinity.models.infinity import Infinity
from infinity.models.basic import *
import PIL.Image as PImage
from torchvision.transforms.functional import to_tensor
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

import matplotlib.pyplot as plt
import seaborn as sns


def extract_key_val(text):
    pattern = r'<(.+?):(.+?)>'
    matches = re.findall(pattern, text)
    key_val = {}
    for match in matches:
        key_val[match[0]] = match[1].lstrip()
    return key_val

def encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt=False):
    if enable_positive_prompt:
        print(f'before positive_prompt aug: {prompt}')
        prompt = aug_with_positive_prompt(prompt)
        print(f'after positive_prompt aug: {prompt}')
    print(f'prompt={prompt}')
    captions = [prompt]
    tokens = text_tokenizer(text=captions, max_length=512, padding='max_length', truncation=True, return_tensors='pt')  # todo: put this into dataset
    input_ids = tokens.input_ids.cuda(non_blocking=True)
    mask = tokens.attention_mask.cuda(non_blocking=True)
    text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
    lens: List[int] = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    Ltext = max(lens)    
    kv_compact = []
    for len_i, feat_i in zip(lens, text_features.unbind(0)):
        kv_compact.append(feat_i[:len_i])
    kv_compact = torch.cat(kv_compact, dim=0)
    text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    return text_cond_tuple

def aug_with_positive_prompt(prompt):
    for key in ['man', 'woman', 'men', 'women', 'boy', 'girl', 'child', 'person', 'human', 'adult', 'teenager', 'employee', 
                'employer', 'worker', 'mother', 'father', 'sister', 'brother', 'grandmother', 'grandfather', 'son', 'daughter']:
        if key in prompt:
            prompt = prompt + '. very smooth faces, good looking faces, face to the camera, perfect facial features'
            break
    return prompt

def enhance_image(image):
    for t in range(1):
        contrast_image = image.copy()
        contrast_enhancer = ImageEnhance.Contrast(contrast_image)
        contrast_image = contrast_enhancer.enhance(1.05)  # 增强对比度
        color_image = contrast_image.copy()
        color_enhancer = ImageEnhance.Color(color_image)
        color_image = color_enhancer.enhance(1.05)  # 增强饱和度
    return color_image

def gen_one_img_with_power(
    infinity_test,
    vae,
    text_tokenizer,
    text_encoder,
    prompt,
    csv_path,                # << 必填：功耗结果输出 CSV 文件
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    power_hz=200.0,          # 采样频率（可调：100~500）
    device_index=0,          # NVML 里的 GPU 索引；单卡映射时一般就是 0
):
    """
    只测功耗，不测时间、不写 Excel、不保存图片。
    会在 csv_path 写入列：stage, block, attn_J, ffn_J, other_J, total_J, seq_len
    """
    # 组 cfg/tau
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)

    # 文本编码，拿到 label_B_or_BLT
    text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt)
    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, negative_prompt)
    else:
        negative_label_B_or_BLT = None

    # 直接调用 Infinity.infer_cfg3_power（你已在 Infinity 里按我上一条加过它）
    infinity_test.infer_cfg3_power(
        csv_path=csv_path,
        vae=vae,
        scale_schedule=scale_schedule,
        label_B_or_BLT=text_cond_tuple,
        B=1,
        negative_label_B_or_BLT=negative_label_B_or_BLT,
        g_seed=g_seed,
        cfg_sc=cfg_sc,
        cfg_list=cfg_list,
        tau_list=tau_list,
        top_k=top_k,
        top_p=top_p,
        returns_vemb=1,              # 只测能量，不用返回图像向量
        gumbel=gumbel,
        norm_cfg=False,
        cfg_exp_k=cfg_exp_k,
        cfg_insertion_layer=cfg_insertion_layer,
        vae_type=vae_type,
        softmax_merge_topk=softmax_merge_topk,
        ret_img=False,               # 不返图
        trunk_scale=1000,
        gt_leak=gt_leak,
        gt_ls_Bl=gt_ls_Bl,
        inference_mode=True,
        sampling_per_bits=sampling_per_bits,
        power_hz=power_hz,
        device_index=device_index,
    )
    print(f"[power] Wrote per-stage×block energy CSV: {csv_path}")


def gen_one_img_with_time(
    infinity_test, 
    vae, 
    text_tokenizer,
    text_encoder,
    prompt, 
    excel_path,                 # <<< 新增：Excel 输出路径，如 "/path/to/profiling.xlsx"
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
):
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)  # 确保目录存在

    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)

    text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt)
    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, negative_prompt)
    else:
        negative_label_B_or_BLT = None

    print(f'cfg: {cfg_list}, tau: {tau_list}')
    with torch.amp.autocast('cuda',enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        stt = time.time()
        # 关键改动：调用 infer_cfg2，并把 excel_path 传进去
        _, _, img_list = infinity_test.infer_cfg2(
            excel_path=excel_path,
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
            B=1, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
            cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
            returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
            cfg_exp_k=cfg_exp_k, cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type, softmax_merge_topk=softmax_merge_topk,
            ret_img=True, trunk_scale=1000,
            gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
            sampling_per_bits=sampling_per_bits,
        )
    print(f"cost: {time.time() - sstt:.3f}s, infinity cost={time.time() - stt:.3f}s")
    img = img_list[0]
    return img

def analyze_attention_maps(attention_data, save_path=None):
    """分析收集到的attention map数据"""
    
    if save_path:
        # 确保保存路径存在，如果不存在则创建
        # exist_ok=True 避免目录已存在时抛出错误
        os.makedirs(save_path, exist_ok=True)
        print(f"Save directory created/checked: {save_path}")
    
    analysis_results = {}
    
    for stage_key, stage_data in attention_data.items():
        print(f"Analyzing {stage_key}...")
            
        for attn_type, attn_map in stage_data.items():
            if attn_map is not None:
                # 基础统计分析
                analysis = {
                    'shape': attn_map.shape,
                    'mean': attn_map.mean().item(),
                    'std': attn_map.std().item(),
                    'max': attn_map.max().item(),
                    'min': attn_map.min().item(),
                }
                
                analysis_results[f"{stage_key}_{attn_type}"] = analysis
                
                # 可视化（可选）
                if save_path:   
                    if attn_map.dim() == 4:# B, H, L, L
                    # 取第一个batch和第一个head的可视化
                        sample_attn = attn_map[0, 0].numpy()
                    elif attn_map.dim() == 3: # H, L, L
                    # 取第一个head的可视化
                        sample_attn = attn_map[0].numpy()
                        
                    ratio=len(sample_attn)/len(sample_attn[0]) #y/x
                    
                    plt.figure(figsize=(10, 8*ratio))
                    sns.heatmap(sample_attn, cmap='Blues', 
                               xticklabels=False, yticklabels=False)
                    plt.title(f'{stage_key}_{attn_type}')
                    plt.savefig(f'{save_path}/{stage_key}_{attn_type}.png')
                    plt.close()
    
    return analysis_results

def gen_one_img(
    infinity_test, 
    vae, 
    text_tokenizer,
    text_encoder,
    prompt, 
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    save_sparsity_masks=False,  # 新增参数
    sparsity_mask_save_path=None,  # 新增参数
):
    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt)
    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, negative_prompt)
    else:
        negative_label_B_or_BLT = None
    print(f'cfg: {cfg_list}, tau: {tau_list}')
    with torch.amp.autocast('cuda',enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        stt = time.time()
        _, _, img_list = infinity_test.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
            B=1, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
            cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
            returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
            cfg_exp_k=cfg_exp_k, cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type, softmax_merge_topk=softmax_merge_topk,
            ret_img=True, trunk_scale=1000,
            gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
            sampling_per_bits=sampling_per_bits, save_sparsity_masks=save_sparsity_masks, 
            sparsity_mask_save_path=sparsity_mask_save_path
        )
    print(f"cost: {time.time() - sstt}, infinity cost={time.time() - stt}")
    img = img_list[0]
    return img


def gen_one_img_with_attnmap(
    infinity_test, 
    vae, 
    text_tokenizer,
    text_encoder,
    prompt, 
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    attention_stages=[1, 2, 3, 4, 5],
    save_path='./attention_analysis',
):
    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt)
    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, negative_prompt)
    else:
        negative_label_B_or_BLT = None
    print(f'cfg: {cfg_list}, tau: {tau_list}')
    with torch.amp.autocast('cuda',enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        stt = time.time()
        _, _, img_list, attn_maps = infinity_test.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
            B=1, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
            cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
            returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
            cfg_exp_k=cfg_exp_k, cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type, softmax_merge_topk=softmax_merge_topk,
            ret_img=True, trunk_scale=1000,
            gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
            sampling_per_bits=sampling_per_bits,
            collect_attention_maps=True,  # 新增参数：是否收集attention map
            attention_stages=attention_stages, 
        )
    print(f"cost: {time.time() - sstt}, infinity cost={time.time() - stt}")
    img = img_list[0]
    if attn_maps:
        analysis = analyze_attention_maps(attn_maps, save_path=save_path)
        print(f"Attention analysis completed and maps saved at {save_path}")
    return img, analysis

def get_prompt_id(prompt):
    md5 = hashlib.md5()
    md5.update(prompt.encode('utf-8'))
    prompt_id = md5.hexdigest()
    return prompt_id

def save_slim_model(infinity_model_path, save_file=None, device='cpu', key='gpt_fsdp'):
    print('[Save slim model]')
    full_ckpt = torch.load(infinity_model_path, map_location=device)
    infinity_slim = full_ckpt['trainer'][key]
    # ema_state_dict = cpu_d['trainer'].get('gpt_ema_fsdp', state_dict)
    if not save_file:
        save_file = osp.splitext(infinity_model_path)[0] + '-slim.pth'
    print(f'Save to {save_file}')
    torch.save(infinity_slim, save_file)
    print('[Save slim model] done')
    return save_file

def load_tokenizer(t5_path =''):
    print(f'[Loading tokenizer and text encoder]')
    text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(t5_path, revision=None, legacy=True)
    text_tokenizer.model_max_length = 512
    text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.float16)
    text_encoder.to('cuda')
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    return text_tokenizer, text_encoder

def load_infinity(
    rope2d_each_sa_layer, 
    rope2d_normalized_by_hw, 
    use_scale_schedule_embedding, 
    pn, 
    use_bit_label, 
    add_lvl_embeding_only_first_block, 
    model_path='', 
    scale_schedule=None, 
    vae=None, 
    device='cuda', 
    model_kwargs=None,
    text_channels=2048,
    apply_spatial_patchify=0,
    use_flex_attn=False,
    bf16=False,
    checkpoint_type='torch',
    customized_flash_attn=False,
    sparsity_ratio_sa=None,
    sparsity_ratio_ca=None
):
    print(f'[Loading Infinity]')
    text_maxlen = 512
    with torch.amp.autocast('cuda',enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        infinity_test: Infinity = Infinity(
            vae_local=vae, text_channels=text_channels, text_maxlen=text_maxlen,
            shared_aln=True, raw_scale_schedule=scale_schedule,
            checkpointing='full-block',
            customized_flash_attn=customized_flash_attn,
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=use_flex_attn,
            add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
            use_bit_label=use_bit_label,
            rope2d_each_sa_layer=rope2d_each_sa_layer,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            pn=pn,
            apply_spatial_patchify=apply_spatial_patchify,
            inference_mode=True,
            train_h_div_w_list=[1.0],
            sparsity_ratio_sa=sparsity_ratio_sa,
            sparsity_ratio_ca=sparsity_ratio_ca,
            **model_kwargs,
        ).to(device=device)
        print(f'[you selected Infinity with {model_kwargs=}] model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={bf16}')

        if bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()

        infinity_test.eval()
        infinity_test.requires_grad_(False)

        infinity_test.cuda()
        torch.cuda.empty_cache()

        print(f'[Load Infinity weights]')
        if checkpoint_type == 'torch':
            state_dict = torch.load(model_path, map_location=device)
            print(infinity_test.load_state_dict(state_dict))
        elif checkpoint_type == 'torch_shard':
            from transformers.modeling_utils import load_sharded_checkpoint
            load_sharded_checkpoint(infinity_test, model_path, strict=False)
        infinity_test.rng = torch.Generator(device=device)
        return infinity_test

def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    return im.add(im).add_(-1)

def joint_vi_vae_encode_decode(vae, image_path, scale_schedule, device, tgt_h, tgt_w):
    pil_image = Image.open(image_path).convert('RGB')
    inp = transform(pil_image, tgt_h, tgt_w)
    inp = inp.unsqueeze(0).to(device)
    scale_schedule = [(item[0], item[1], item[2]) for item in scale_schedule]
    t1 = time.time()
    h, z, _, all_bit_indices, _, infinity_input = vae.encode(inp, scale_schedule=scale_schedule)
    t2 = time.time()
    recons_img = vae.decode(z)[0]
    if len(recons_img.shape) == 4:
        recons_img = recons_img.squeeze(1)
    print(f'recons: z.shape: {z.shape}, recons_img shape: {recons_img.shape}')
    t3 = time.time()
    print(f'vae encode takes {t2-t1:.2f}s, decode takes {t3-t2:.2f}s')
    recons_img = (recons_img + 1) / 2
    recons_img = recons_img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)
    gt_img = (inp[0] + 1) / 2
    gt_img = gt_img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)
    print(recons_img.shape, gt_img.shape)
    return gt_img, recons_img, all_bit_indices

def load_visual_tokenizer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load vae
    if args.vae_type in [14,16,18,20,24,32,64]:
        from infinity.models.bsq_vae.vae import vae_model
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type
        codebook_size = 2**codebook_dim
        if args.apply_spatial_patchify:
            patch_size = 8
            encoder_ch_mult=[1, 2, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult=[1, 2, 4, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4, 4]
        vae = vae_model(args.vae_path, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size, 
                        encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(device)
    else:
        raise ValueError(f'vae_type={args.vae_type} not supported')
    return vae

def load_transformer(vae, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path
    if args.checkpoint_type == 'torch': 
        # copy large model to local; save slim to local; and copy slim to nas; load local slim model
        if osp.exists(args.cache_dir):
            local_model_path = osp.join(args.cache_dir, 'tmp', model_path.replace('/', '_'))
        else:
            local_model_path = model_path
        if args.enable_model_cache:
            slim_model_path = model_path.replace('ar-', 'slim-')
            local_slim_model_path = local_model_path.replace('ar-', 'slim-')
            os.makedirs(osp.dirname(local_slim_model_path), exist_ok=True)
            print(f'model_path: {model_path}, slim_model_path: {slim_model_path}')
            print(f'local_model_path: {local_model_path}, local_slim_model_path: {local_slim_model_path}')
            if not osp.exists(local_slim_model_path):
                if osp.exists(slim_model_path):
                    print(f'copy {slim_model_path} to {local_slim_model_path}')
                    shutil.copyfile(slim_model_path, local_slim_model_path)
                else:
                    if not osp.exists(local_model_path):
                        print(f'copy {model_path} to {local_model_path}')
                        shutil.copyfile(model_path, local_model_path)
                    save_slim_model(local_model_path, save_file=local_slim_model_path, device=device)
                    print(f'copy {local_slim_model_path} to {slim_model_path}')
                    if not osp.exists(slim_model_path):
                        shutil.copyfile(local_slim_model_path, slim_model_path)
                        os.remove(local_model_path)
                        os.remove(model_path)
            slim_model_path = local_slim_model_path
        else:
            slim_model_path = model_path
        print(f'load checkpoint from {slim_model_path}')
    elif args.checkpoint_type == 'torch_shard':
        slim_model_path = model_path

    if args.model_type == 'infinity_2b':
        kwargs_model = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8) # 2b model
    elif args.model_type == 'infinity_8b':
        kwargs_model = dict(depth=40, embed_dim=3584, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    elif args.model_type == 'infinity_layer12':
        kwargs_model = dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer16':
        kwargs_model = dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer24':
        kwargs_model = dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer32':
        kwargs_model = dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer40':
        kwargs_model = dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer48':
        kwargs_model = dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    infinity = load_infinity(
        rope2d_each_sa_layer=args.rope2d_each_sa_layer, 
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        use_scale_schedule_embedding=args.use_scale_schedule_embedding,
        pn=args.pn,
        use_bit_label=args.use_bit_label, 
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block, 
        model_path=slim_model_path, 
        scale_schedule=None, 
        vae=vae, 
        device=device, 
        model_kwargs=kwargs_model,
        text_channels=args.text_channels,
        apply_spatial_patchify=args.apply_spatial_patchify,
        use_flex_attn=args.use_flex_attn,
        bf16=args.bf16,
        checkpoint_type=args.checkpoint_type,
        customized_flash_attn=args.customized_flash_attn,
        sparsity_ratio_sa=args.sparsity_ratio_sa,
        sparsity_ratio_ca=args.sparsity_ratio_ca
    )
    return infinity

def add_common_arguments(parser):
    parser.add_argument('--cfg', type=str, default='3')
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--pn', type=str, required=True, choices=['0.06M', '0.25M', '1M'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cfg_insertion_layer', type=int, default=0)
    parser.add_argument('--vae_type', type=int, default=1)
    parser.add_argument('--vae_path', type=str, default='')
    parser.add_argument('--add_lvl_embeding_only_first_block', type=int, default=0, choices=[0,1])
    parser.add_argument('--use_bit_label', type=int, default=1, choices=[0,1])
    parser.add_argument('--model_type', type=str, default='infinity_2b')
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1, choices=[0,1])
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2, choices=[0,1,2])
    parser.add_argument('--use_scale_schedule_embedding', type=int, default=0, choices=[0,1])
    parser.add_argument('--sampling_per_bits', type=int, default=1, choices=[1,2,4,8,16])
    parser.add_argument('--text_encoder_ckpt', type=str, default='')
    parser.add_argument('--text_channels', type=int, default=2048)
    parser.add_argument('--apply_spatial_patchify', type=int, default=0, choices=[0,1])
    parser.add_argument('--h_div_w_template', type=float, default=1.000)
    parser.add_argument('--use_flex_attn', type=int, default=0, choices=[0,1])
    parser.add_argument('--enable_positive_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--cache_dir', type=str, default='/dev/shm')
    parser.add_argument('--enable_model_cache', type=int, default=0, choices=[0,1])
    parser.add_argument('--checkpoint_type', type=str, default='torch')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bf16', type=int, default=1, choices=[0,1])
    parser.add_argument('--customized_flash_attn', type=bool, default=False)
    parser.add_argument('--sparsity_ratio_sa', type=float, default=None)
    parser.add_argument('--sparsity_ratio_ca', type=float, default=None)
    if not any(opt.option_strings == ['--enable_model_cache'] for opt in parser._actions):
        parser.add_argument("--enable_model_cache", action="store_true", default=True, help="Enable local model cache.")

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--prompt', type=str, default='a dog')
    parser.add_argument('--save_file', type=str, default='./tmp.jpg')
    args = parser.parse_args()

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    
    # load text encoder
    text_tokenizer, text_encoder = load_tokenizer(t5_path =args.text_encoder_ckpt)
    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)
    
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [ (1, h, w) for (_, h, w) in scale_schedule]

    with torch.amp.autocast('cuda',dtype=torch.bfloat16):
        with torch.no_grad():
            generated_image = gen_one_img(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                args.prompt,
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
            )
    os.makedirs(osp.dirname(osp.abspath(args.save_file)), exist_ok=True)
    cv2.imwrite(args.save_file, generated_image.cpu().numpy())
    print(f'Save to {osp.abspath(args.save_file)}')
