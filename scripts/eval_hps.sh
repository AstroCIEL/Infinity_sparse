#!/bin/bash

infer_eval_hpsv21() {
    # ${pip_ext} install hpsv2
    # ${pip_ext}install -U diffusers
    # sudo apt install python3-tk
    # wget https://dl.fbaipublicfiles.com/mmf/clip/bpe_simple_vocab_16e6.txt.gz
    # mv bpe_simple_vocab_16e6.txt.gz /root/miniconda3/envs/infinity/lib/python3.10/site-packages/hpsv2/src/open_clip

    mkdir -p ${out_dir}
    ${python_ext} tools/eval_hpsv2.py \
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
    --outdir ${out_dir}/images | tee ${out_dir}/log.txt
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


# HPS v2.1
out_dir=${out_dir_root}/hpsv21_${sub_fix}
infer_eval_hpsv21
