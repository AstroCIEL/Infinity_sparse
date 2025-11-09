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
sparsity_ratio_sa=0.7
long_caption_fid=0
coco30k_prompts=0
total_prompts=666

# 路径配置
infinity_model_path=/root/autodl-tmp/Predata/infinity_2b_reg.pth
vae_path=/root/autodl-tmp/Predata/infinity_vae_d32reg.pth
text_encoder_ckpt=/root/autodl-tmp/Predata/flan-t5-xl
jsonl_filepath=/root/autodl-tmp/Predata/coco2014_val_prompts_full.jsonl

# 输出目录
out_dir_root=/root/autodl-tmp/Infinity_sparse/output/clip_evaluation
generation_dir=${out_dir_root}/generated_images_$(date +%Y%m%d_%H%M%S)

# 阶段2：CLIP配置
clip_model_name="ViT-B/32"
clip_batch_size=64
clip_gpu_id=0  # 可以使用不同的GPU进行CLIP评估

echo "[INFO] Generation directory: ${generation_dir}"
mkdir -p "${generation_dir}"

#############################
# 阶段1：多GPU图像生成
#############################
echo "=== STAGE 1: Multi-GPU Image Generation ==="

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "[INFO] Found ${NUM_GPUS} GPUs for generation"

# 启动生成进程
GEN_PID_FILE="${generation_dir}/generation_pids.txt"
> "$GEN_PID_FILE"

for ((RANK=0; RANK<NUM_GPUS; RANK++)); do
    echo "[INFO] Launching generation on GPU ${RANK}"
    
    CUDA_VISIBLE_DEVICES=${RANK} python tools/infer_clip_score.py \
        --cfg ${cfg} \
        --tau ${tau} \
        --pn ${pn} \
        --model_path ${infinity_model_path} \
        --vae_type ${vae_type} \
        --vae_path ${vae_path} \
        --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
        --use_bit_label ${use_bit_label} \
        --model_type ${model_type} \
        --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
        --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
        --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
        --checkpoint_type ${checkpoint_type} \
        --text_encoder_ckpt ${text_encoder_ckpt} \
        --text_channels ${text_channels} \
        --apply_spatial_patchify ${apply_spatial_patchify} \
        --cfg_insertion_layer ${cfg_insertion_layer} \
        --coco30k_prompts ${coco30k_prompts} \
        --jsonl_filepath ${jsonl_filepath} \
        --long_caption_fid ${long_caption_fid} \
        --out_dir ${generation_dir} \
        --rank ${RANK} \
        --world_size ${NUM_GPUS} \
        --total_prompts ${total_prompts} \
        --sparsity_ratio_sa ${sparsity_ratio_sa} \
        > "${generation_dir}/gpu${RANK}.log" 2>&1 &
    
    echo $! >> "$GEN_PID_FILE"
    echo "[INFO] GPU ${RANK} generation started with PID $!"
done

# 等待生成完成
echo "[INFO] Waiting for image generation to complete..."
GEN_PIDS=()
while IFS= read -r pid; do
    if [ -n "$pid" ]; then
        GEN_PIDS+=("$pid")
    fi
done < "$GEN_PID_FILE"

for pid in "${GEN_PIDS[@]}"; do
    if ps -p "$pid" > /dev/null; then
        wait "$pid"
    fi
done

echo "[INFO] Image generation completed"

# 检查生成结果
generated_count=$(find "${generation_dir}/generated_images" -name "*.jpg" 2>/dev/null | wc -l || echo "0")
echo "[INFO] Generated ${generated_count} images"

if [ "$generated_count" -eq 0 ]; then
    echo "[ERROR] No images generated. Please check the logs."
    exit 1
fi

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