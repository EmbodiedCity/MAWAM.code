"""CUDA-graphed streaming causal video VAE wrapper (`T=1`, `T=4`, `T=8` transitions)."""

from __future__ import annotations

import torch


class ThreeStateGraphedVAE:
    """Pre-captures CUDA Graphs for `T=1` (init), `T=4` (first chunk), and `T=8` (steady chunks)."""

    def __init__(self, stream_wrapper, device: torch.device, h: int, w: int, dtype=torch.bfloat16) -> None:
        self.wrapper = stream_wrapper
        self.device = device
        self.dtype = dtype
        self.h = h
        self.w = w
        self.in_steady = False
        self._build_all_graphs()

    def clear_cache(self) -> None:
        self.in_steady = False

    def _build_all_graphs(self) -> None:
        with torch.cuda.device(self.device):
            self.static_in1 = torch.zeros((1, 3, 1, self.h, self.w), device=self.device, dtype=self.dtype)
            self.static_in4 = torch.zeros((1, 3, 4, self.h, self.w), device=self.device, dtype=self.dtype)
            self.static_in8 = torch.zeros((1, 3, 8, self.h, self.w), device=self.device, dtype=self.dtype)
            s_stream = torch.cuda.Stream(device=self.device)

            # 1. T=1 (None -> State A)
            with torch.cuda.stream(s_stream):
                self.wrapper.clear_cache()
                _ = self.wrapper.encode_chunk(self.static_in1)
            torch.cuda.synchronize(self.device)
            state_a = [c.clone() if c is not None else None for c in self.wrapper.feat_cache]
            self.cache_a = [c.clone() if c is not None else None for c in state_a]

            self.graph_init1 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph_init1, stream=s_stream, capture_error_mode="thread_local"):
                self.wrapper.clear_cache()
                self.out_init1 = self.wrapper.encode_chunk(self.static_in1)
                for ca, nc in zip(self.cache_a, self.wrapper.feat_cache):
                    if ca is not None and nc is not None:
                        ca.copy_(nc)
            torch.cuda.synchronize(self.device)

            # 2. T=4 (State A -> State B)
            with torch.cuda.stream(s_stream):
                self.wrapper.feat_cache = [c.clone() if c is not None else None for c in self.cache_a]
                _ = self.wrapper.encode_chunk(self.static_in4)
            torch.cuda.synchronize(self.device)
            state_b = [c.clone() if c is not None else None for c in self.wrapper.feat_cache]
            self.cache_b = [c.clone() if c is not None else None for c in state_b]

            self.graph_first4 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph_first4, stream=s_stream, capture_error_mode="thread_local"):
                self.wrapper.feat_cache = list(self.cache_a)
                self.out_first4 = self.wrapper.encode_chunk(self.static_in4)
                for cb, nc in zip(self.cache_b, self.wrapper.feat_cache):
                    if cb is not None and nc is not None:
                        cb.copy_(nc)
            torch.cuda.synchronize(self.device)

            # 3. T=8 (State B -> State B)
            with torch.cuda.stream(s_stream):
                self.wrapper.feat_cache = [c.clone() if c is not None else None for c in state_b]
                _ = self.wrapper.encode_chunk(self.static_in8)
            torch.cuda.synchronize(self.device)

            self.graph_steady8 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph_steady8, stream=s_stream, capture_error_mode="thread_local"):
                self.wrapper.feat_cache = list(self.cache_b)
                self.out_steady8 = self.wrapper.encode_chunk(self.static_in8)
                for cb, nc in zip(self.cache_b, self.wrapper.feat_cache):
                    if cb is not None and nc is not None:
                        cb.copy_(nc)
            torch.cuda.synchronize(self.device)

    def encode_async(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[2]
        with torch.cuda.device(self.device):
            if t == 1:
                self.static_in1.copy_(x)
                self.graph_init1.replay()
                self.in_steady = False
                return self.out_init1
            if t == 4 and not self.in_steady:
                self.static_in4.copy_(x)
                self.graph_first4.replay()
                self.in_steady = True
                return self.out_first4
            if t == 8 and self.in_steady:
                self.static_in8.copy_(x)
                self.graph_steady8.replay()
                return self.out_steady8
            raise ValueError(f"Unexpected frame count T={t} (in_steady={self.in_steady})")
