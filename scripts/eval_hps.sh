#!/bin/bash

set -euo pipefail

eval_hpsv21_infer() {
    # ${pip_ext} install hpsv2
    # ${pip_ext}install -U diffusers
    # sudo apt install python3-tk
    # wget https://dl.fbaipublicfiles.com/mmf/clip/bpe_simple_vocab_16e6.txt.gz
    # mv bpe_simple_vocab_16e6.txt.gz /root/miniconda3/envs/infinity/lib/python3.10/site-packages/hpsv2/src/open_clip

    mkdir -p ${out_dir}
    
    # 获取GPU数量
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    echo "[INFO] Found ${NUM_GPUS} GPUs"

    # 创建进程ID文件
    PID_FILE="/tmp/hpsv2_pids.txt"
    > "$PID_FILE"

    echo "[INFO] Starting parallel generation on ${NUM_GPUS} GPUs..."

    # 启动所有GPU进程
    for ((RANK=0; RANK<NUM_GPUS; RANK++)); do
        echo "[INFO] Launching process for GPU ${RANK}"
        
        # 启动进程
        CUDA_VISIBLE_DEVICES=${RANK} ${python_ext} tools/eval_hpsv2_multi_gpu.py \
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
            --sparsity_ratio_sa ${sparsity_ratio_sa} \
            --rank ${RANK} \
            --world_size ${NUM_GPUS} \
            --outdir ${out_dir}/images > "${out_dir}/log_rank${RANK}.txt" 2>&1 &
        
        # 记录进程ID
        echo $! >> "$PID_FILE"
        echo "[INFO] GPU ${RANK} started with PID $!"
    done

    # 等待所有进程完成
    echo "[INFO] Waiting for all processes to complete..."

    PIDS=()
    while IFS= read -r pid; do
        if [ -n "$pid" ]; then
            PIDS+=("$pid")
        fi
    done < "$PID_FILE"

    for pid in "${PIDS[@]}"; do
        if ps -p "$pid" > /dev/null; then
            echo "[INFO] Waiting for PID $pid to complete..."
            wait "$pid"
            echo "[INFO] PID $pid completed"
        fi
    done

    echo "[INFO] All GPU processes have completed"

    # 检查完成标记文件
    for ((RANK=0; RANK<NUM_GPUS; RANK++)); do
        COMPLETION_FILE="${out_dir}/images/rank${RANK}.done"
        if [ -f "$COMPLETION_FILE" ]; then
            echo "[INFO] GPU ${RANK} completion status:"
            cat "$COMPLETION_FILE"
        else
            echo "[WARNING] GPU ${RANK}: Completion file not found - $COMPLETION_FILE"
        fi
    done

    # 检查生成的图片数量
    IMAGE_COUNT=$(find ${out_dir}/images -name "*.jpg" | wc -l)
    echo "[INFO] Total images generated: $IMAGE_COUNT"

}

eval_hpsv21() {
    # ${pip_ext} install hpsv2
    # ${pip_ext}install -U diffusers
    # sudo apt install python3-tk
    # wget https://dl.fbaipublicfiles.com/mmf/clip/bpe_simple_vocab_16e6.txt.gz
    # mv bpe_simple_vocab_16e6.txt.gz /home/tiger/.local/lib/python3.9/site-packages/hpsv2/src/open_clip

    #### asume you have correct env

    ${python_ext} tools/eval_hpsv2.py \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --eval 1 \
    --checkpoint_path ${checkpoint_path} \
    --outdir ${out_dir}/images | tee ${out_dir}/log_eval.txt
}

python_ext=python3
pip_ext=pip3

# set arguments for inference
pn=1M
model_type=infinity_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
out_dir_root=/root/autodl-tmp/Infinity_sparse/output/hpsv2
vae_type=32
cfg=4
tau=1
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_channels=2048
apply_spatial_patchify=0
cfg_insertion_layer=0
sub_fix=cfg${cfg}_tau${tau}_cfg_insertion_layer${cfg_insertion_layer}
sparsity_ratio_sa=0.75

infinity_model_path=/root/autodl-tmp/Predata/infinity_2b_reg.pth
vae_path=/root/autodl-tmp/Predata/infinity_vae_d32reg.pth
text_encoder_ckpt=/root/autodl-tmp/Predata/flan-t5-xl
checkpoint_path=/root/autodl-tmp/Infinity_sparse/evaluation/hpsv2/HPS_v2.1_compressed_4.pt
## download at https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt

# HPS v2.1
out_dir=${out_dir_root}/hpsv21_${sub_fix}
# you may need to generate images in A env
eval_hpsv21_infer
# evaluate in B env
eval_hpsv21
