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

# 采样超参（保持和你之前一致或按需修改）
cfg=4
tau=1

# 模型与权重路径（按你机器实况改）
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

# —— 你的权重路径（按你给的路径填好）——
infinity_model_path=/DISK1/home/yx_zhao31/Infinity/Predata/infinity_2b_reg.pth
vae_path=/DISK1/home/yx_zhao31/Infinity/Predata/infinity_vae_d32reg.pth
text_encoder_ckpt=/DISK1/home/yx_zhao31/Infinity/Predata/flan-t5-xl

# —— 你的 jsonl（短 caption 版本）——
jsonl_filepath=/DISK1/home/yx_zhao31/Infinity/Predata/coco2014_val_prompts_full.jsonl
# 因为上面 jsonl 只有 text + image_path，没有 long_caption，所以置 0
long_caption_fid=0
# 不用 COCO30k 内置 prompt
coco30k_prompts=0

# 输出目录
out_dir_root=/DISK1/home/yx_zhao31/Infinity/output/infinity_fid
sub_fix=cfg${cfg}_tau${tau}_cfg_insertion_layer${cfg_insertion_layer}
out_dir=${out_dir_root}/val_fid_${sub_fix}

echo "[INFO] jsonl: ${jsonl_filepath}"
echo "[INFO] out_dir: ${out_dir}"
rm -rf "${out_dir}"
mkdir -p "${out_dir}"

#############################
# Step 1: generate images for FID
#############################
${python_ext} tools/comprehensive_infer.py \
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
  --out_dir ${out_dir}

# 运行完后，会得到：
# ${out_dir}/pred    -> 生成图
# ${out_dir}/gt      -> 对应的真实图拷贝/链接
# ${out_dir}/meta*   -> 元信息

#############################
# Step 2: compute FID
#############################
${python_ext} tools/fid_score.py \
  ${out_dir}/pred \
  ${out_dir}/gt | tee ${out_dir}/fid_log.txt

echo "[DONE] FID finished. See ${out_dir}/fid_log.txt"
