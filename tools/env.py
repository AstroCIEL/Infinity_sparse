# 用与你出错时同一个 python 可执行程序
/DISK1/home/yx_zhao31/.conda/envs/infinity/bin/python - <<'PY'
import sys, torch, os
print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch.cuda:", torch.version.cuda, "is_available:", torch.cuda.is_available())
print("site-packages:", next(p for p in sys.path if p.endswith("site-packages")))
PY
