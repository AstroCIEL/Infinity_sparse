#!/usr/bin/env bash
# 可选：打开调试
# set -x
set -euo pipefail

############################################
# 0) 路径配置（按需修改）
############################################
INF_ROOT="/DISK1/home/yx_zhao31/Infinity"
OUT_ROOT="/DISK1/home/yx_zhao31/Infinity/output/infinity_2b_evaluation"

INFINITY_MODEL_PATH="/DISK1/home/yx_zhao31/Infinity/Predata/infinity_2b_reg.pth"
VAE_TYPE="32"
VAE_PATH="/DISK1/home/yx_zhao31/Infinity/Predata/infinity_vae_d32.pth"
TEXT_ENCODER_CKPT="/DISK1/home/yx_zhao31/Infinity/Predata/flan-t5-xl"

# 重要：必须是具体的 .py 配置文件
MODEL_CONFIG="${INF_ROOT}/evaluation/gen_eval/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
# 重要：建议直接给 .pth 文件路径（或给目录 + --options model=...）
MODEL_PTH="/DISK1/home/yx_zhao31/Infinity/Predata/geneval/object/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"

# Conda 环境名（按你机器上的名字改）
ENV_INFER="infinity"  # 生成图片用
ENV_EVAL="geneval"    # 评测用

############################################
# 1) 生成参数（与你原来一致）
############################################
PN="1M"
MODEL_TYPE="infinity_2b"
USE_SCALE_SCHEDULE_EMBEDDING=0
USE_BIT_LABEL=1
CHECKPOINT_TYPE="torch"
CFG=4
TAU=1
ROPE2D_NORMALIZED_BY_HW=2
ADD_LVL_EMB_ONLY_FIRST_BLOCK=1
ROPE2D_EACH_SA_LAYER=1
TEXT_CHANNELS=2048
APPLY_SPATIAL_PATCHIFY=0
CFG_INSERTION_LAYER=0
REWRITE_PROMPT=0

SUB_FIX="cfg${CFG}_tau${TAU}_cfg_insertion_layer${CFG_INSERTION_LAYER}"
OUTDIR="${OUT_ROOT}/gen_eval_${SUB_FIX}_rewrite_prompt${REWRITE_PROMPT}"
IMG_OUT="${OUTDIR}/images"
RES_DIR="${OUTDIR}/results"
META_FILE="${OUTDIR}/metadata.jsonl"   # infer4eval.py 通常会写在 OUTDIR 根目录

############################################
# 2) 小工具
############################################
ensure_file() { [[ -f "$1" ]] || { echo "[ERR] file not found: $1"; exit 1; }; }
ensure_dir()  { [[ -d "$1" ]] || { echo "[ERR] dir  not found: $1"; exit 1; }; }

# 更稳健的环境激活：在触发 conda 钩子前关闭 `-u`
activate_env() {
  local env_name="$1"
  set +u
  # 优先用 conda 提供的 base 路径；失败就尝试常见路径
  local conda_base
  conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -z "${conda_base}" || ! -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
    # 可按需添加其它备选路径
    for guess in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
      if [[ -f "${guess}/etc/profile.d/conda.sh" ]]; then
        conda_base="${guess}"
        break
      fi
    done
  fi
  if ! source "${conda_base}/etc/profile.d/conda.sh" 2>/dev/null; then
    echo "[ERR] failed to source conda.sh (base=${conda_base})"; exit 1
  fi
  conda activate "${env_name}"
  set -u
}

# 安全退出当前 conda 环境（关闭 `-u`，避免钩子脚本里未定义变量）
safe_deactivate() {
  set +u
  conda deactivate >/dev/null 2>&1 || true
  set -u
}

# 无论退出还是出错都清一次环境，避免残留
trap 'safe_deactivate' EXIT

############################################
# 3) 生成 GenEval 图片（infinity 环境）
############################################
gen_eval_generate() {
  mkdir -p "${IMG_OUT}"
  # 显式确保当前没有遗留环境
  safe_deactivate
  activate_env "${ENV_INFER}"

  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export HF_HUB_OFFLINE=1

  LOGF="${OUTDIR}/infer.log"
  mkdir -p "${OUTDIR}"
  echo "[INFO] Logging to ${LOGF}"

  stdbuf -oL -eL python -u "${INF_ROOT}/evaluation/gen_eval/infer4eval.py" \
    --cfg "${CFG}" \
    --tau "${TAU}" \
    --pn "${PN}" \
    --model_path "${INFINITY_MODEL_PATH}" \
    --vae_type "${VAE_TYPE}" \
    --vae_path "${VAE_PATH}" \
    --add_lvl_embeding_only_first_block "${ADD_LVL_EMB_ONLY_FIRST_BLOCK}" \
    --use_bit_label "${USE_BIT_LABEL}" \
    --model_type "${MODEL_TYPE}" \
    --rope2d_each_sa_layer "${ROPE2D_EACH_SA_LAYER}" \
    --rope2d_normalized_by_hw "${ROPE2D_NORMALIZED_BY_HW}" \
    --use_scale_schedule_embedding "${USE_SCALE_SCHEDULE_EMBEDDING}" \
    --checkpoint_type "${CHECKPOINT_TYPE}" \
    --text_encoder_ckpt "${TEXT_ENCODER_CKPT}" \
    --text_channels "${TEXT_CHANNELS}" \
    --apply_spatial_patchify "${APPLY_SPATIAL_PATCHIFY}" \
    --cfg_insertion_layer "${CFG_INSERTION_LAYER}" \
    --outdir "${IMG_OUT}" \
    --rewrite_prompt "${REWRITE_PROMPT}" \
    2>&1 | tee -a "${LOGF}"

  echo "[OK] Images generated at: ${IMG_OUT}"
  [[ -f "${META_FILE}" ]] || echo "[WARN] ${META_FILE} not found; if missing, pass your own --metadata."
}

############################################
# 4) 目标检测评测与汇总（geneval 环境）
############################################
gen_eval_detect_and_score() {
  mkdir -p "${RES_DIR}"
  ensure_dir  "${IMG_OUT}"
  ensure_file "${MODEL_CONFIG}"
  ensure_file "${MODEL_PTH}"

  # 切换前先清当前环境，随后进入评测环境
  safe_deactivate
  activate_env "${ENV_EVAL}"

  # metadata 参数（存在就带上）
  local -a META_ARG=()
  if [[ -f "${META_FILE}" ]]; then
    META_ARG=(--metadata "${META_FILE}")
  else
    echo "[WARN] ${META_FILE} not found, continue without --metadata (需要每个子目录有 metadata.jsonl/.json)"
  fi

  stdbuf -oL -eL python -u "${INF_ROOT}/evaluation/gen_eval/evaluate_images.py" "${IMG_OUT}" \
    --outfile "${RES_DIR}/det.jsonl" \
    --model-config "${MODEL_CONFIG}" \
    --model-path "${MODEL_PTH}" \
    "${META_ARG[@]}"

  # 汇总
  stdbuf -oL -eL python -u "${INF_ROOT}/evaluation/gen_eval/summary_scores.py" "${RES_DIR}/det.jsonl" > "${RES_DIR}/res.txt" || true
  echo "================ GenEval Scores ================"
  cat "${RES_DIR}/res.txt" || true
  echo "==============================================="
}

############################################
# 5) 主流程
############################################
main() {
  gen_eval_generate
  gen_eval_detect_and_score
  echo "[DONE] Results saved to: ${RES_DIR}"
}

main "$@"
