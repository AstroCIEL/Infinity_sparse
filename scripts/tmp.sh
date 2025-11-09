#!/bin/bash
# scripts/run_clip_evaluation_two_stage.sh

set -euo pipefail

#############################
# 用户配置
#############################
# 阶段1：生成配置
pn=1M
cfg=4
tau=1
n_samples=1
total_prompts=1000

# 模型与权重路径
model_type=infinity_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
vae_type=32
apply_spatial_patchify=0
cfg_insertion_layer=0
rope2d_each_sa_layer=1
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
text_channels=2048
sparsity_ratio_sa=0.6
long_caption_fid=0
coco30k_prompts=0
total_prompts=999

# 路径配置
infinity_model_path=/root/autodl-tmp/Predata/infinity_2b_reg.pth
vae_path=/root/autodl-tmp/Predata/infinity_vae_d32reg.pth
text_encoder_ckpt=/root/autodl-tmp/Predata/flan-t5-xl
jsonl_filepath=/root/autodl-tmp/Predata/coco2014_val_prompts_full.jsonl

# 输出目录
out_dir_root=/root/autodl-tmp/Infinity_sparse/output/clip_evaluation
generation_dir=${out_dir_root}/generated_images_20251109_205452

# 阶段2：CLIP配置
clip_model_name="ViT-B/32"
clip_batch_size=64
clip_gpu_id=0  # 可以使用不同的GPU进行CLIP评估

echo "[INFO] Generation directory: ${generation_dir}"
#############################
# 阶段2：CLIP评分
#############################
echo ""
echo "=== STAGE 2: CLIP Score Evaluation ==="

# 等待一段时间确保所有文件写入完成
sleep 5

# 使用指定的GPU进行CLIP评估
echo "[INFO] Starting CLIP evaluation on GPU ${clip_gpu_id}"

CUDA_VISIBLE_DEVICES=${clip_gpu_id} python tools/compute_clip_score.py \
    --generation_dir ${generation_dir} \
    --clip_model_name ${clip_model_name} \
    --batch_size ${clip_batch_size} \
    --gpu_id ${clip_gpu_id} \
    > "${generation_dir}/clip_evaluation.log" 2>&1

echo "[INFO] CLIP evaluation completed"

# 显示结果
if [ -f "${generation_dir}/clip_score_statistics.txt" ]; then
    echo ""
    echo "=== FINAL RESULTS ==="
    cat "${generation_dir}/clip_score_statistics.txt"
else
    echo "[WARNING] Results file not found. Check ${generation_dir}/clip_evaluation.log for errors."
fi

echo "[DONE] Two-stage CLIP evaluation completed successfully."