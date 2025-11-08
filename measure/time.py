# -*- coding: utf-8 -*-
# 完整版：带 warmup + 计时写 Excel
# 依赖：tools/run_infinity.py 内已实现 infer_cfg2 与 gen_one_img_with_time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"   # <<< 一定要在 import torch 之前

import random
import argparse
import time
import os
import os.path as osp
import sys
import cv2
import numpy as np

import torch

# 可选：让 cudnn 按需选择更快算法（首次会稍慢，之后更快）
torch.backends.cudnn.benchmark = True

# 由于上面把可见 GPU 映射为单卡，这里 set_device(0) 对应的是那张卡
torch.cuda.set_device(0)

# 需要包含 infer_cfg2 与 gen_one_img_with_time、get_prompt_id、各加载函数
from tools.run_infinity import *  

print("python:", sys.executable)
print("cwd:", os.getcwd())
print("torch version:", torch.__version__)
print("cuda:", torch.version.cuda, "is_available:", torch.cuda.is_available())


# ======================
# Warmup 封装
# ======================
def warmup_infinity(
    infinity_test,
    vae,
    text_tokenizer,
    text_encoder,
    scale_schedule,
    cfg,
    tau,
    cfg_insertion_layer,
    vae_type,
    sampling_per_bits,
    rounds=2,
    use_full_schedule=True,
    prompt="warmup prompt",
    seed_base=1234,
):
    """
    说明：
    - rounds: 热身轮数（1~2 通常够用）
    - use_full_schedule=False：默认仅用 schedule 的前 1/3 触发编译（更快）。若要更稳，改 True。
    - 不落盘、不写 Excel，仅触发编译、缓存、KV 初始化等。
    """
    warmup_schedule = scale_schedule if use_full_schedule else scale_schedule[:max(1, len(scale_schedule)//3)]
    print(f"[Warmup] rounds={rounds}, stages={len(warmup_schedule)} / {len(scale_schedule)}")

    torch.cuda.synchronize()
    t0 = time.time()

    for i in range(rounds):
        _ = gen_one_img(
            infinity_test=infinity_test,
            vae=vae,
            text_tokenizer=text_tokenizer,
            text_encoder=text_encoder,
            prompt=prompt,
            g_seed=seed_base + i,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=cfg,
            tau_list=tau,
            scale_schedule=warmup_schedule,
            cfg_insertion_layer=[cfg_insertion_layer],
            vae_type=vae_type,
            sampling_per_bits=sampling_per_bits,
            enable_positive_prompt=0,
        )

    torch.cuda.synchronize()
    print(f"[Warmup] done in {time.time() - t0:.3f}s")


# ======================
# 路径配置
# ======================
model_path='/DISK1/home/yx_zhao31/Infinity/Predata/infinity_2b_reg.pth'
vae_path='/DISK1/home/yx_zhao31/Infinity/Predata/infinity_vae_d32reg.pth'
text_encoder_ckpt = '/DISK1/home/yx_zhao31/Infinity/Predata/flan-t5-xl'

# ======================
# 组装参数
# ======================
args = argparse.Namespace(
    pn='1M',
    model_path=model_path,
    cfg_insertion_layer=0,
    vae_type=32,
    vae_path=vae_path,
    add_lvl_embeding_only_first_block=1,
    use_bit_label=1,
    model_type='infinity_2b',
    rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2,
    use_scale_schedule_embedding=0,
    sampling_per_bits=1,
    text_encoder_ckpt=text_encoder_ckpt,
    text_channels=2048,
    apply_spatial_patchify=0,
    h_div_w_template=1.000,
    use_flex_attn=0,
    cache_dir='/dev/shm',
    checkpoint_type='torch',
    seed=0,
    bf16=1,
    save_file='tmp.jpg'
)
# 兼容：确保存在 enable_model_cache
for k, v in {"enable_model_cache": True}.items():
    if not hasattr(args, k):
        setattr(args, k, v)

# ======================
# 加载组件
# ======================
# text encoder
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
# vae
vae = load_visual_tokenizer(args)
# transformer
infinity = load_transformer(vae, args)

# ======================
# 生成参数
# ======================
prompt = """A beautiful Chinese woman with graceful features, close-up portrait, long flowing black hair, wearing a traditional silk cheongsam delicately embroidered with floral patterns, face softly illuminated by ambient light, serene expression"""
cfg = 3
tau = 0.5
h_div_w = 1/1  # aspect ratio, height:width
seed = random.randint(0, 10000)
enable_positive_prompt = 0

# scale_schedule
h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

# ======================
# Warmup（可配置）
# ======================
WARMUP_ROUNDS = 2
USE_FULL_SCHEDULE = True  # 如需更稳的热身改 True（更慢）

warmup_infinity(
    infinity_test=infinity,
    vae=vae,
    text_tokenizer=text_tokenizer,
    text_encoder=text_encoder,
    scale_schedule=scale_schedule,
    cfg=cfg,
    tau=tau,
    cfg_insertion_layer=args.cfg_insertion_layer,
    vae_type=args.vae_type,
    sampling_per_bits=args.sampling_per_bits,
    rounds=WARMUP_ROUNDS,
    use_full_schedule=USE_FULL_SCHEDULE,
)

# ======================
# Excel 路径（按 prompt_id+seed 命名）
# ======================
profile_dir = "/DISK1/home/yx_zhao31/Infinity/measure/profiling"
os.makedirs(profile_dir, exist_ok=True)
excel_path = osp.join(profile_dir, f"{get_prompt_id(prompt)}_{seed}.xlsx")

# ======================
# 生成并计时到 Excel
# ======================
torch.cuda.synchronize()
t_run0 = time.time()

generated_image = gen_one_img_with_time(
    infinity_test=infinity,
    vae=vae,
    text_tokenizer=text_tokenizer,
    text_encoder=text_encoder,
    prompt=prompt,
    excel_path=excel_path,          # << 关键：把 Excel 输出路径传进去
    g_seed=seed,
    gt_leak=0,
    gt_ls_Bl=None,
    cfg_list=cfg,
    tau_list=tau,
    scale_schedule=scale_schedule,
    cfg_insertion_layer=[args.cfg_insertion_layer],
    vae_type=args.vae_type,
    sampling_per_bits=args.sampling_per_bits,
    enable_positive_prompt=enable_positive_prompt,
)
torch.cuda.synchronize()
print(f"[Run] done in {time.time() - t_run0:.3f}s")

# ======================
# 保存图片
# ======================
args.save_file = 'ipynb_tmp.jpg'
os.makedirs(osp.dirname(osp.abspath(args.save_file)), exist_ok=True)
cv2.imwrite(args.save_file, generated_image.cpu().numpy())
print(f"Save to {osp.abspath(args.save_file)}")
print(f"Timing Excel saved to: {excel_path}")
