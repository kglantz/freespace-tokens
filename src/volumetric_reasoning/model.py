"""Qwen3-VL decoder reading 3-D scene tokens (surface + free/unknown boxes) in place of image tokens.

Sequence (Qwen3-VL chat format, scene tokens where an image's tokens would be):

    <|im_start|>user\\n<|vision_start|> [scene tokens] <|vision_end|>{prompt}<|im_end|>\\n<|im_start|>assistant\\n
    {answer}<|im_end|>

Scene token embeddings (trainable parts marked *):
    surface: f + A*(f) + phi*(p) + e*_surface + m*  f = pooled frozen Qwen3-VL image feature (already in the
                                                    decoder's embedding space), A = residual MLP,
                                                    m = mark of navigation endpoint A / B (0 for other tokens)
    box:     psi*(size) + phi*(center) + e*_type  type = free or unknown
phi is shared by all scene tokens. A, phi and psi end in zero-initialized layers and e_surface starts at 0, so at
initialization a surface token is exactly the Qwen visual token of its block (plus its mark embedding if it belongs
to an endpoint; the "no mark" row is fixed at 0); box tokens start as their type embedding (word-embedding scale).
Box extent is encoded in the content (psi), position in phi and in RoPE.

3-D rotary positions follow Qwen3-VL's image convention. An image starting at sequence position s puts merged
patch (row, col) at (t, h, w) = (s, s + row, s + col), and the following text continues at s + max(H, W).
A scene token at metric position (x, y, z) gets (t, h, w) = (s + z/u, s + y/u, s + x/u), with coordinates taken
relative to the scene's minimum and u = ROPE_UNIT_M (one 16 cm block per position, as one patch per position for
images). The following text continues at s + (largest of these offsets) + 1. Positions may be fractional.

The decoder is frozen apart from LoRA adapters on its attention projections. The loss is cross-entropy on the
answer tokens and the closing <|im_end|>; the output layer is applied only at those positions.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .geometry.labels import FREE, UNKNOWN

ROPE_UNIT_M = 0.16
PREFIX = "<|im_start|>user\n<|vision_start|>"
SUFFIX = "<|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
END = "<|im_end|>"


class Fourier(nn.Module):
    """3-D metric vector -> sin/cos at wavelengths from 32 m down to 8 cm (+ the raw values) -> MLP."""

    def __init__(self, hidden: int, bands: int = 16, width: int = 512):
        super().__init__()
        wavelengths = 32.0 * (0.08 / 32.0) ** (torch.arange(bands) / (bands - 1))
        self.register_buffer("freq", 2 * math.pi / wavelengths, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(3 + 6 * bands, width), nn.GELU(), nn.Linear(width, hidden))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, v):  # (N, 3) meters
        a = (v[:, :, None] * self.freq).flatten(1)
        return self.mlp(torch.cat([v, a.sin(), a.cos()], dim=1))


class SceneEncoder(nn.Module):
    TYPE_INDEX = {FREE: 1, UNKNOWN: 2}  # 0 = surface

    def __init__(self, hidden: int, feature_dim: int, word_std: float, adapter_width: int = 256):
        super().__init__()
        if feature_dim != hidden:
            raise ValueError("Features must come from the vision encoder of the same Qwen3-VL model")
        self.adapter = nn.Sequential(nn.Linear(feature_dim, adapter_width), nn.GELU(), nn.Linear(adapter_width, hidden))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.position = Fourier(hidden)
        self.size = Fourier(hidden)
        self.kind = nn.Embedding(3, hidden)
        nn.init.normal_(self.kind.weight, std=word_std)
        self.mark = nn.Embedding(3, hidden, padding_idx=0)  # 0 = no mark (stays 0), 1 = endpoint A, 2 = endpoint B
        nn.init.normal_(self.mark.weight, std=word_std)
        with torch.no_grad():
            self.kind.weight[0].zero_()
            self.mark.weight[0].zero_()

    def forward(self, ex) -> torch.Tensor:
        """(Ns + Nb, hidden) embeddings: surface tokens, then boxes (order of the example)."""
        f = ex["surface_feature"].float()
        surface = f + self.adapter(f) + self.position(ex["surface_position"]) + self.kind.weight[0]
        if "mark" in ex:
            surface = surface + self.mark(ex["mark"])
        t = torch.zeros_like(ex["box_type"])
        for label, index in self.TYPE_INDEX.items():
            t[ex["box_type"] == label] = index
        boxes = self.size(ex["box_size"]) + self.position(ex["box_center"]) + self.kind(t)
        return torch.cat([surface, boxes])


def scene_rope_positions(positions: torch.Tensor, start: int, unit: float = ROPE_UNIT_M):
    """(3, N) rotary positions (t, h, w) = start + (z, y, x) relative to the minimum, in units; next text start."""
    rel = (positions - positions.min(0).values) / unit
    return start + rel[:, [2, 1, 0]].T, start + float(rel.max()) + 1


class FreeSpaceVLM(nn.Module):
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-4B-Instruct", lora_rank: int = 16, lora_alpha: float = 32,
                 lora_dropout: float = 0.05, attn_implementation: str = "sdpa"):
        super().__init__()
        from peft import LoraConfig, inject_adapter_in_model
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        qwen = Qwen3VLForConditionalGeneration.from_pretrained(model_name, dtype=torch.bfloat16,
                                                               attn_implementation=attn_implementation)
        qwen.requires_grad_(False)
        self.decoder = qwen.model.language_model  # the vision encoder is not needed: features are precomputed
        self.lm_head = qwen.lm_head
        inject_adapter_in_model(LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]), self.decoder)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        cfg = qwen.config
        embed = self.decoder.embed_tokens.weight
        self.scene_encoder = SceneEncoder(cfg.text_config.hidden_size, cfg.vision_config.out_hidden_size,
                                          embed.float().std().item())
        self.end_id = self.tokenizer.convert_tokens_to_ids(END)
        self._prefix_ids = self._ids(PREFIX)

    def _ids(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def trainable_state_dict(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.named_parameters() if v.requires_grad}

    def load_trainable_state_dict(self, state: dict) -> None:
        trainable = {k for k, v in self.named_parameters() if v.requires_grad}
        if trainable != set(state):
            raise KeyError(f"Checkpoint does not match the trainable tensors: missing {sorted(trainable - set(state))[:3]}, "
                           f"unexpected {sorted(set(state) - trainable)[:3]}")
        self.load_state_dict(state, strict=False)

    # --- inputs ---------------------------------------------------------------------
    def inputs(self, ex, answer: bool):
        """inputs_embeds (1, L, H), position_ids (3, 1, L), number of answer tokens at the end (0 if no answer)."""
        dev = self.lm_head.weight.device
        ex = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in ex.items()}
        suffix = self._ids(SUFFIX.format(prompt=ex["prompt"]))
        target = self._ids(ex["answer"]) + [self.end_id] if answer else []
        scene = self.scene_encoder(ex).to(torch.bfloat16)
        embed = self.decoder.embed_tokens
        prefix_ids = torch.tensor(self._prefix_ids, device=dev)
        text_ids = torch.tensor(suffix + target, device=dev)
        x = torch.cat([embed(prefix_ids), scene, embed(text_ids)])[None]

        s = len(self._prefix_ids)
        xyz = torch.cat([ex["surface_position"], ex["box_center"]]).float()
        scene_pos, next_pos = scene_rope_positions(xyz, s)
        prefix_pos = torch.arange(s, device=dev, dtype=torch.float32).expand(3, -1)
        text_pos = (next_pos + torch.arange(len(text_ids), device=dev, dtype=torch.float32)).expand(3, -1)
        pos = torch.cat([prefix_pos, scene_pos, text_pos], dim=1)[:, None]
        return x, pos, len(target)

    # --- training -------------------------------------------------------------------
    def loss(self, ex) -> torch.Tensor:
        x, pos, n = self.inputs(ex, answer=True)
        h = self.decoder(inputs_embeds=x, position_ids=pos, use_cache=False).last_hidden_state
        logits = self.lm_head(h[0, -n - 1:-1]).float()  # positions that predict the answer tokens
        target = torch.tensor(self._ids(ex["answer"]) + [self.end_id], device=logits.device)
        return F.cross_entropy(logits, target)

    # --- inference ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, ex, max_new_tokens: int = 24) -> str:
        """Greedy decoding with the KV cache; text positions continue one per token."""
        x, pos, _ = self.inputs(ex, answer=False)
        out = self.decoder(inputs_embeds=x, position_ids=pos, use_cache=True)
        cache, last = out.past_key_values, pos[0, 0, -1]
        tokens = []
        for step in range(max_new_tokens):
            token = int(self.lm_head(out.last_hidden_state[0, -1]).argmax())
            if token == self.end_id:
                break
            tokens.append(token)
            p = (last + 1 + step).view(1, 1, 1).expand(3, 1, 1)
            out = self.decoder(inputs_embeds=self.decoder.embed_tokens(torch.tensor([[token]], device=x.device)),
                               position_ids=p, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
