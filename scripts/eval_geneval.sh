#!/bin/bash

test_gen_eval_infer() {

    #############################
    # 获取GPU数量
    #############################
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    echo "[INFO] Found ${NUM_GPUS} GPUs"

    #############################
    # Step 1: 预处理prompt重写（单GPU执行）
    #############################
    # echo "[INFO] Preprocessing prompt rewriting..."
    # CUDA_VISIBLE_DEVICES=0 ${python_ext} /root/autodl-tmp/Infinity_sparse/tools/infer4eval_geneval.py \
    #     --cfg ${cfg} \
    #     --tau ${tau} \
    #     --pn ${pn} \
    #     --model_path ${infinity_model_path} \
    #     --vae_type ${vae_type} \
    #     --vae_path ${vae_path} \
    #     --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    #     --use_bit_label ${use_bit_label} \
    #     --model_type ${model_type} \
    #     --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    #     --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    #     --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    #     --checkpoint_type ${checkpoint_type} \
    #     --text_encoder_ckpt ${text_encoder_ckpt} \
    #     --text_channels ${text_channels} \
    #     --apply_spatial_patchify ${apply_spatial_patchify} \
    #     --cfg_insertion_layer ${cfg_insertion_layer} \
    #     --outdir ${out_dir}/images \
    #     --sparsity_ratio_sa ${sparsity_ratio_sa} \
    #     --metadata_file ${metadata_file} \
    #     --rewrite_prompt ${rewrite_prompt} \
    #     --load_rewrite_prompt_cache 0 \
    #     --n_samples 0 \
    #     --rank 0 \
    #     --world_size 1 \
    #     --total_prompts 1  # 只处理一个prompt来生成缓存

    # echo "[INFO] Prompt rewriting preprocessing completed"

    #############################
    # Step 2: 多GPU并行生成图片
    #############################
    # 创建进程ID文件
    PID_FILE="/tmp/geneval_pids.txt"
    > "$PID_FILE"

    echo "[INFO] Starting parallel generation on ${NUM_GPUS} GPUs..."

    # 启动所有GPU进程
    for ((RANK=0; RANK<NUM_GPUS; RANK++)); do
        echo "[INFO] Launching process for GPU ${RANK}"
        
        # 直接启动进程
        CUDA_VISIBLE_DEVICES=${RANK} ${python_ext} tools/infer4eval_geneval.py \
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
            --outdir ${out_dir}/images \
            --sparsity_ratio_sa ${sparsity_ratio_sa} \
            --metadata_file ${metadata_file} \
            --rewrite_prompt ${rewrite_prompt} \
            --load_rewrite_prompt_cache 0 \
            --n_samples ${n_samples} \
            --rank ${RANK} \
            --world_size ${NUM_GPUS} \
            --total_prompts ${total_prompts} \
            > "${out_dir}/gpu${RANK}.log" 2>&1 &
        
        # 记录进程ID
        echo $! >> "$PID_FILE"
        echo "[INFO] GPU ${RANK} started with PID $!"
    done

    #############################
    # Step 3: 等待所有进程完成
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
    # Step 4: 清理临时文件
    #############################
    echo "[INFO] Cleaning up temporary files..."
    rm -f "$PID_FILE"
    
    # 删除GPU完成标记文件
    for ((RANK=0; RANK<NUM_GPUS; RANK++)); do
        completion_file="${out_dir}/images/gpu${RANK}.done"
        if [ -f "$completion_file" ]; then
            rm -f "$completion_file"
        fi
    done

    echo "[INFO] Cleanup completed"
}

test_gen_eval() {
    # ${pip_ext} install -U openmim
    # mim install mmengine mmcv-full==1.7.2
    # ${pip_ext} install mmdet==2.28.2 pytorch_lightning clip_benchmark open-clip-torch==2.20.0
    # ${pip_ext} install -U diffusers
    # sudo apt install libgl1
    # ${pip_ext} install openai
    # ${pip_ext} install httpx==0.20.0
    
    ####assume you have correct env
    ####download https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth -O "$1/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
    ####and mv to weights/mask2former and rename as mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth

    # detect objects
    ${python_ext} evaluation/gen_eval/evaluate_images.py ${out_dir}/images \
    --outfile ${out_dir}/results/det.jsonl \
    --model-config evaluation/gen_eval/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py \
    --model-path weights/mask2former

    # accumulate results
    ${python_ext} evaluation/gen_eval/summary_scores.py ${out_dir}/results/det.jsonl > ${out_dir}/results/res.txt
    cat ${out_dir}/results/res.txt
}


python_ext=python3
pip_ext=pip3

# set arguments for inference
pn=1M
model_type=infinity_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
out_dir_root=/root/autodl-tmp/Infinity_sparse/output/geneval
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
metadata_file=evaluation/gen_eval/prompts/evaluation_metadata.jsonl

# 添加新参数
n_samples=5  # 每个prompt生成的样本数
total_prompts=-1  # 处理的prompt总数，-1表示全部

infinity_model_path=/root/autodl-tmp/Predata/infinity_2b_reg.pth
vae_path=/root/autodl-tmp/Predata/infinity_vae_d32reg.pth
text_encoder_ckpt=/root/autodl-tmp/Predata/flan-t5-xl

# GenEval
rewrite_prompt=0
out_dir=${out_dir_root}/gen_eval_${sub_fix}_rewrite_prompt${rewrite_prompt}_round2_real_rewrite

# 创建输出目录
mkdir -p ${out_dir}

echo "[INFO] Starting multi-GPU GenEval evaluation..."
echo "[INFO] Output directory: ${out_dir}"
echo "[INFO] Number of samples per prompt: ${n_samples}"
echo "[INFO] Total prompts to process: ${total_prompts}"

# maybe you should first generate pics in A env
test_gen_eval_infer
# then you evalutae them in B env
test_gen_eval

echo "[DONE] GenEval evaluation completed."