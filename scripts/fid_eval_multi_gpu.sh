#!/usr/bin/env bash
set -euo pipefail

#############################
# Minimal deps for FID only
#############################
python_ext=python3
pip_ext=pip3
${pip_ext} install -U pytorch_fid

#############################
# User config (edit here)
#############################
# 分辨率选择：0.06M≈256、0.25M≈512、1M≈1024
pn=1M

# 采样超参
cfg=4
tau=1

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
#sparsity_ratio_sa=0.6
long_caption_fid=0
coco30k_prompts=0
total_prompts=9999

# 路径配置
infinity_model_path=/root/autodl-tmp/Predata/infinity_2b_reg.pth
vae_path=/root/autodl-tmp/Predata/infinity_vae_d32reg.pth
text_encoder_ckpt=/root/autodl-tmp/Predata/flan-t5-xl
jsonl_filepath=/root/autodl-tmp/Predata/coco2014_val_prompts_full.jsonl

# 输出目录
out_dir_root=/root/autodl-tmp/Infinity_sparse/output/infinity_fid
sub_fix=cfg${cfg}_tau${tau}_cfg_insertion_layer${cfg_insertion_layer}
out_dir=${out_dir_root}/val_fid_${sub_fix}

echo "[INFO] jsonl: ${jsonl_filepath}"
echo "[INFO] out_dir: ${out_dir}"
rm -rf "${out_dir}"
mkdir -p "${out_dir}"

#############################
# 获取GPU数量
#############################
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "[INFO] Found ${NUM_GPUS} GPUs"

#############################
# Step 1: 多GPU并行生成图片 - 修复版本
#############################
# 创建进程ID文件
PID_FILE="/tmp/infinity_pids.txt"
> "$PID_FILE"

echo "[INFO] Starting parallel generation on ${NUM_GPUS} GPUs..."

# 启动所有GPU进程
for ((RANK=0; RANK<NUM_GPUS; RANK++)); do
    echo "[INFO] Launching process for GPU ${RANK}"
    
    # 直接启动进程，不使用子shell
    CUDA_VISIBLE_DEVICES=${RANK} ${python_ext} tools/comprehensive_infer.py \
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
        --save4fid_eval 1 \
        --jsonl_filepath ${jsonl_filepath} \
        --long_caption_fid ${long_caption_fid} \
        --out_dir ${out_dir} \
        --rank ${RANK} \
        --world_size ${NUM_GPUS} \
        --total_prompts ${total_prompts} \
        > "${out_dir}/gpu${RANK}.log" 2>&1 &
    
    # 记录进程ID
    echo $! >> "$PID_FILE"
    echo "[INFO] GPU ${RANK} started with PID $!"
done

#############################
# Step 2: 等待所有进程完成 - 修复版本
#############################
echo "[INFO] Waiting for all processes to complete..."

# 读取所有PID并等待
PIDS=()
while IFS= read -r pid; do
    if [ -n "$pid" ]; then
        PIDS+=("$pid")
    fi
done < "$PID_FILE"

# 等待所有进程完成
for pid in "${PIDS[@]}"; do
    if ps -p "$pid" > /dev/null; then
        echo "[INFO] Waiting for PID $pid to complete..."
        wait "$pid"
        echo "[INFO] PID $pid completed"
    fi
done

echo "[INFO] All GPU processes have completed"

#############################
# Step 3: 检查输出文件并合并元数据
#############################
echo "[INFO] Checking output files..."

# 检查每个GPU是否生成了元数据文件
for ((RANK=0; RANK<NUM_GPUS; RANK++)); do
    META_FILE="${out_dir}/meta_info_gpu${RANK}.jsonl"
    if [ -f "$META_FILE" ]; then
        LINE_COUNT=$(wc -l < "$META_FILE" 2>/dev/null || echo "0")
        echo "[INFO] GPU ${RANK}: $LINE_COUNT lines in metadata"
    else
        echo "[WARNING] GPU ${RANK}: Metadata file not found - $META_FILE"
    fi
done

# 合并元数据文件
echo "[INFO] Merging metadata files..."
COMBINED_META="${out_dir}/meta_info_combined.jsonl"
> "$COMBINED_META"  # 清空文件

for ((RANK=0; RANK<NUM_GPUS; RANK++)); do
    META_FILE="${out_dir}/meta_info_gpu${RANK}.jsonl"
    if [ -f "$META_FILE" ] && [ -s "$META_FILE" ]; then
        cat "$META_FILE" >> "$COMBINED_META"
        echo "[INFO] Added GPU ${RANK} metadata to combined file"
    fi
done

COMBINED_COUNT=$(wc -l < "$COMBINED_META" 2>/dev/null || echo "0")
echo "[INFO] Combined metadata has $COMBINED_COUNT lines"

#############################
# Step 4: FID calculation
#############################
${python_ext} tools/fid_score.py \
    ${out_dir}/pred \
    ${out_dir}/gt | tee ${out_dir}/fid_log.txt

echo "[DONE] Comparison completed."