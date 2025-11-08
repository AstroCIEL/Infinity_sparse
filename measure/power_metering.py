# -*- coding: utf-8 -*-
# 修正版：按 block 计总能量（带 padding），子模块（attn/ffn）不加 padding，避免重叠双算；
# 修复 pre/post hook 时间戳存取的闭包变量问题；补齐 CSV 导出依赖与目录处理。

import os
import time
import threading
import bisect
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

# ===== PowerMeter: 轮询 NVML，积分能量（J） =====
class PowerMeter:
    """
    以固定频率采样 GPU 功率（W），并做时间积分得到能量（J）。
    - device_index: NVML 的 GPU index
    - hz: 采样频率（建议 20~50Hz；若层很短促，可尝试 100Hz）
    用法：
        m = PowerMeter(0, 50); m.start(); ... ; E = m.energy_between(t0, t1); m.stop()
    """
    def __init__(self, device_index: int = 0, hz: float = 50.0):
        try:
            import pynvml
            self._pynvml = pynvml
        except Exception as e:
            raise RuntimeError(
                "pynvml 未安装或不可用，请先 `pip install nvidia-ml-py` 并确保 NVIDIA 驱动正常。"
            ) from e
        self.dev_i = device_index
        self.hz = float(max(1.0, hz))
        self.dt = 1.0 / self.hz
        self._stop = threading.Event()
        self._t = []      # 绝对时间戳（秒, time.time()）
        self._w = []      # 功率（W）
        self._lock = threading.Lock()
        self._th = None
        self._handle = None

    def start(self, warmup_ms: float = 150.0):
        pn = self._pynvml
        pn.nvmlInit()
        self._handle = pn.nvmlDeviceGetHandleByIndex(self.dev_i)
        self._stop.clear()
        self._t.clear(); self._w.clear()

        def _loop():
            # 连续轮询；NVML 自身可能做了短窗平均
            while not self._stop.is_set():
                try:
                    p_mw = pn.nvmlDeviceGetPowerUsage(self._handle)  # 毫瓦
                    p_w = float(p_mw) / 1000.0
                except Exception:
                    p_w = 0.0
                now = time.time()
                with self._lock:
                    self._t.append(now)
                    self._w.append(p_w)
                time.sleep(self.dt)

        self._th = threading.Thread(target=_loop, daemon=True)
        self._th.start()
        # 预热：避免第一些样本异常
        time.sleep(max(0.0, warmup_ms / 1000.0))

    def stop(self):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=1.0)
            self._th = None
        if self._handle is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
            self._handle = None

    def now(self) -> float:
        """当前时间（秒，time.time()）。"""
        return time.time()

    def energy_between(self, t0: float, t1: float) -> float:
        """
        计算 [t0, t1] 时间窗的能量（焦耳）。
        用线性插值 + 梯形积分。若窗外超出采样范围，会自动 clamp。
        """
        if t1 <= t0:
            return 0.0
        with self._lock:
            if not self._t:
                return 0.0
            T = np.asarray(self._t, dtype=np.float64)
            W = np.asarray(self._w, dtype=np.float64)

        lo = max(t0, float(T[0]))
        hi = min(t1, float(T[-1]))
        if hi <= lo:
            return 0.0

        i0 = max(0, bisect.bisect_left(T, lo) - 1)
        i1 = min(len(T) - 1, bisect.bisect_right(T, hi))

        t_seg = [lo]
        w_seg = []
        # 左端插值
        if T[i0] == lo:
            w_seg.append(W[i0])
        else:
            t_left, t_right = T[i0], T[i0+1]
            w_left, w_right = W[i0], W[i0+1]
            alpha = (lo - t_left) / max(1e-9, (t_right - t_left))
            w_lo = (1 - alpha) * w_left + alpha * w_right
            w_seg.append(w_lo)

        # 中间点
        for j in range(i0 + 1, i1):
            t_seg.append(float(T[j]))
            w_seg.append(float(W[j]))

        # 右端插值
        t_seg.append(hi)
        if T[i1] == hi:
            w_seg.append(W[i1])
        else:
            t_left, t_right = T[i1-1], T[i1]
            w_left, w_right = W[i1-1], W[i1]
            alpha = (hi - t_left) / max(1e-9, (t_right - t_left))
            w_hi = (1 - alpha) * w_left + alpha * w_right
            w_seg.append(w_hi)

        t_seg = np.asarray(t_seg, dtype=np.float64)
        w_seg = np.asarray(w_seg, dtype=np.float64)
        e = np.trapz(w_seg, t_seg)        # W * s = J
        return float(max(0.0, e))
