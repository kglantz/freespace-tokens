"""Frozen Qwen3-VL image features, mean-pooled per surface block.

Each RGB frame is resized to IMAGE_WIDTH x IMAGE_HEIGHT (the VSI-Bench video resolution) and encoded by
the vision encoder of the Qwen3-VL model that will be fine-tuned. Its output tokens (`pooler_output`, the
embeddings the language model receives for an image) cover 32 x 32 pixel patches in row-major order:
token `row * GRID_W + col` covers pixels [32 row, 32 row + 32) x [32 col, 32 col + 32).

Every valid depth pixel is back-projected to 3-D; if it lies in a surface block, the feature of the
image patch containing that pixel is added to the block. Depth and RGB cover the same field of view,
so depth pixel (u, v) maps to image position ((u + 0.5) * IMAGE_WIDTH / 256, (v + 0.5) * IMAGE_HEIGHT / 192).
A block's feature is the mean over all its depth pixels in all frames (patches weighted by how many of
the block's pixels they contain).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .geometry.backprojection import depth_to_points

IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
PATCH = 32  # pixels per output token side (16-pixel ViT patches, 2 x 2 merged)
GRID_W, GRID_H = IMAGE_WIDTH // PATCH, IMAGE_HEIGHT // PATCH


class VisionEncoder:
    """The frozen vision encoder (incl. merger) of a Qwen3-VL checkpoint."""

    def __init__(self, model_name: str, device: str):
        from transformers import AutoImageProcessor, Qwen3VLForConditionalGeneration
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, dtype=torch.bfloat16)
        self.visual = model.model.visual.to(device, dtype).eval()
        del model  # only the vision encoder is needed
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.device = device
        self.dim = self.visual.config.out_hidden_size

    @torch.no_grad()
    def encode(self, rgb: np.ndarray) -> np.ndarray:
        """(GRID_H * GRID_W, dim) float32 features of one RGB uint8 image."""
        image = cv2.resize(rgb, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        n = IMAGE_WIDTH * IMAGE_HEIGHT  # fixes the size: no further resizing by the processor
        inputs = self.processor(images=[image], size={"shortest_edge": n, "longest_edge": n}, return_tensors="pt")
        grid = inputs["image_grid_thw"]
        assert grid.tolist() == [[1, GRID_H * 2, GRID_W * 2]], grid
        pixels = inputs["pixel_values"].to(self.device, self.visual.dtype)
        tokens = self.visual(pixels, grid_thw=grid.to(self.device), return_dict=True).pooler_output
        assert tokens.shape == (GRID_H * GRID_W, self.dim), tokens.shape
        return tokens.float().cpu().numpy()


def depth_pixel_patches(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Image-patch index of every valid depth pixel (same row-major order as `depth_to_points`)."""
    v, u = np.nonzero(mask)
    h, w = depth.shape
    col = np.floor((u + 0.5) * IMAGE_WIDTH / w / PATCH).astype(np.int64)
    row = np.floor((v + 0.5) * IMAGE_HEIGHT / h / PATCH).astype(np.int64)
    return row * GRID_W + col


def pool_block_features(frames, encoder: VisionEncoder, surface_block: np.ndarray, origin: np.ndarray,
                        block_size_m: float, shape_blocks, min_depth: float = 0.1, max_depth: float = 4.0):
    """Mean feature (Ns, dim) of every surface block and its number of contributing depth pixels (Ns,)."""
    index = np.full(tuple(shape_blocks), -1, np.int64)  # block -> surface-token index
    index[tuple(surface_block.T)] = np.arange(len(surface_block))
    sums = torch.zeros(len(surface_block), encoder.dim, dtype=torch.float64)
    counts = np.zeros(len(surface_block), np.int64)
    for f in frames:
        feats = torch.from_numpy(encoder.encode(f.rgb)).double()
        points, mask = depth_to_points(f.depth, f.K_depth, f.cam_to_world, min_depth, max_depth)
        patch = depth_pixel_patches(f.depth, mask)
        block = np.floor((points - origin) / block_size_m).astype(np.int64)
        inside = ((block >= 0) & (block < np.array(shape_blocks))).all(1)
        token = np.full(len(points), -1, np.int64)
        token[inside] = index[tuple(block[inside].T)]
        keep = token >= 0
        sums.index_add_(0, torch.from_numpy(token[keep]), feats[torch.from_numpy(patch[keep])])
        counts += np.bincount(token[keep], minlength=len(surface_block))
    mean = sums / torch.from_numpy(np.maximum(counts, 1))[:, None]
    return mean.float().numpy(), counts
