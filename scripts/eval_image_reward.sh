#!/bin/bash

infer_eval_image_reward() {
    # ${pip_ext} install image-reward pytorch_lightning
    # ${pip_ext} install -U timm diffusers
    # ${pip_ext} install openai==1.34.0 
    # ${pip_ext} install httpx==0.20.0 

    # step 1, infer images
    ${python_ext} tools/infer4eval.py \
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
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --sparsity_ratio_sa ${sparsity_ratio_sa} \
    --metadata_file ${metadata_file} \
    --outdir  ${out_dir}

    # step 2, compute image reward
    # ${pip_ext} install diffusers==0.16.0
    # ${pip_ext} install git+https://github.com/openai/CLIP.git ftfy
    # ${python_ext} evaluation/image_reward/cal_imagereward.py \
    # --meta_file ${out_dir}/metadata.jsonl
}



python_ext=python3
pip_ext=pip3

# set arguments for inference
pn=1M
model_type=infinity_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
out_dir_root=/root/autodl-tmp/Infinity_sparse/output/imagereward_evaluation
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
metadata_file=/root/autodl-tmp/Infinity_sparse/evaluation/image_reward/benchmark-prompts.json

infinity_model_path=/root/autodl-tmp/Predata/infinity_2b_reg.pth
vae_path=/root/autodl-tmp/Predata/infinity_vae_d32reg.pth
text_encoder_ckpt=/root/autodl-tmp/Predata/flan-t5-xl

# ImageReward
out_dir=${out_dir_root}/image_reward_${sub_fix}
infer_eval_image_reward

