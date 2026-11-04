"""Qwen3-VL decoder with proposal surface/volume inputs and metric 3-D MRoPE.

Pinned to Transformers 4.57.1. The stock text decoder accepts four position rows:
one sequential row for causal masking, then three rotary rows. We supply world
x/y/z divided by a fixed metres-per-unit scale for every scene token. Text uses
the same scalar on all three rotary axes. No voxel/cuboid is serialized as text.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

from .common import VARIANTS, write_json


@dataclass
class ModelConfig:
    base_model: str = 'Qwen/Qwen3-VL-4B-Instruct'
    base_revision: str = 'main'
    encoder: str = 'facebook/dinov2-base'
    encoder_revision: str = 'main'
    feature_dim: int = 768
    position_dim: int = 128
    coordinate_scale_m: float = 1.0
    rope_metres_per_unit: float = .02
    lora_rank: int = 16
    lora_alpha: float = 32.
    lora_dropout: float = .05
    variant: str = 'surface_both'
    max_sequence_tokens: int = 262144
    max_answer_tokens: int = 2048
    attention: str = 'sdpa'
    dtype: str = 'bfloat16'


class LoRALinear(nn.Module):
    """Frozen pretrained linear plus trainable (alpha/r) B A residual."""
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        original = self.base(x)
        residual = F.linear(F.linear(self.dropout(x.float()), self.lora_a), self.lora_b)
        return original + (residual * self.scale).to(original.dtype)


class SpatialVLM(nn.Module):
    def __init__(self, config: ModelConfig, device='cuda', gradient_checkpointing=True):
        super().__init__()
        if config.variant not in VARIANTS or config.lora_rank <= 0 or config.rope_metres_per_unit <= 0 or config.coordinate_scale_m <= 0:
            raise ValueError('Invalid model/variant/coordinate settings')
        self.settings, self.device = config, torch.device(device)
        dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[config.dtype]
        if self.device.type == 'cpu':
            dtype = torch.float32
        full = Qwen3VLForConditionalGeneration.from_pretrained(
            config.base_model, revision=config.base_revision, dtype=dtype,
            device_map={'': str(self.device)}, attn_implementation=config.attention,
            low_cpu_mem_usage=True)
        config.base_revision = full.config._commit_hash or config.base_revision
        eos_ids = full.generation_config.eos_token_id
        self.decoder, self.lm_head = full.model.language_model, full.lm_head
        self.decoder.requires_grad_(False)
        self.lm_head.requires_grad_(False)
        # The frozen DINOv2 cache supplies vision features; Qwen's ViT/DeepStack is unused.
        del full
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model, revision=config.base_revision)
        self.stop_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        if self.stop_id is None or self.tokenizer.convert_ids_to_tokens(self.stop_id) != '<|im_end|>':
            raise ValueError('Qwen tokenizer must define <|im_end|>')
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        self.stop_ids = {self.stop_id, self.tokenizer.eos_token_id, *(eos_ids or [])}
        self.hidden_size = int(self.decoder.config.hidden_size)
        self.max_length = min(config.max_sequence_tokens, self.decoder.config.max_position_embeddings)
        for layer in self.decoder.layers:
            for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                setattr(layer.self_attn, name, LoRALinear(getattr(layer.self_attn, name),
                        config.lora_rank, config.lora_alpha, config.lora_dropout))
        self.position = nn.Sequential(nn.Linear(3, config.position_dim), nn.GELU(),
                                      nn.Linear(config.position_dim, config.position_dim)).to(self.device)
        self.feature_norm = nn.LayerNorm(config.feature_dim).to(self.device)
        self.visual_projection = nn.Linear(config.feature_dim + config.position_dim, self.hidden_size).to(self.device)
        self.volume_projection = nn.Linear(config.position_dim + 3, self.hidden_size).to(self.device)
        self.type_embedding = nn.Embedding(3, self.hidden_size).to(self.device)  # visual, free, unknown
        for layer in (self.visual_projection, self.volume_projection):
            nn.init.normal_(layer.weight, std=.02 / math.sqrt(layer.in_features))
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.type_embedding.weight, std=.02)
        if gradient_checkpointing:
            self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.last_sequence_length = 0

    def scene_tokens(self, scene):
        if scene['feature_dim'] != self.settings.feature_dim or scene['encoder'] != self.settings.encoder:
            raise ValueError('Feature cache encoder/dimension differs from model settings')
        if scene['encoder_revision'] != self.settings.encoder_revision:
            raise ValueError('Feature cache encoder revision differs from model settings')
        as_tensor = lambda x: torch.as_tensor(x, device=self.device, dtype=torch.float32)
        xyz = as_tensor(scene['surface_center'])
        features = as_tensor(scene['features'])
        phi = self.position(xyz / self.settings.coordinate_scale_m)
        visual = self.visual_projection(torch.cat([self.feature_norm(features), phi], dim=-1))
        visual = visual + self.type_embedding.weight[0]
        label = scene['region_state']
        variant = self.settings.variant
        keep = ((label == 1) & (variant in ('surface_free', 'surface_both'))
                | (label == 0) & (variant in ('surface_unknown', 'surface_both')))
        region_xyz = as_tensor(scene['region_center'][keep])
        extent = as_tensor(scene['region_extent'][keep])
        region_phi = self.position(region_xyz / self.settings.coordinate_scale_m)
        volume = self.volume_projection(torch.cat([region_phi, extent], dim=-1))
        types = torch.as_tensor(np.where(label[keep] == 1, 1, 2), device=self.device, dtype=torch.long)
        volume = volume + self.type_embedding(types)
        tokens = torch.cat([visual, volume], dim=0)
        coords = torch.cat([xyz, region_xyz], dim=0) / self.settings.rope_metres_per_unit
        # Stable world-coordinate ordering applies to every ablation. No sampling/truncation.
        all_xyz = np.concatenate([scene['surface_center'], scene['region_center'][keep]])
        all_types = np.r_[np.zeros(len(xyz), int), np.where(label[keep] == 1, 1, 2)]
        order = np.lexsort((all_types, all_xyz[:, 2], all_xyz[:, 1], all_xyz[:, 0]))
        order = torch.as_tensor(order, device=self.device)
        return tokens[order].to(self.decoder.embed_tokens.weight.dtype), coords[order]

    def prompt_parts(self, question, history=()):
        marker = '<<FREESPACE_SCENE_EMBEDDINGS>>'
        if marker in question or any(marker in turn['content'] for turn in history):
            raise ValueError('Reserved scene marker in question')
        messages = [
            {'role': 'system', 'content': 'Answer spatial questions using the provided RGB-D scene representation. '
             'Coordinates and extents are in metres in one world frame. Free means observed empty; '
             'unknown means not established free or occupied. Answer concisely.'},
        ]
        turns = [*history, {'role': 'user', 'content': question}]
        for i, turn in enumerate(turns):
            content = turn['content']
            if turn['role'] == 'user':
                content = 'Question: ' + content
            if i == 0:
                content = 'Scene:\n' + marker + '\n' + content
            messages.append({'role': turn['role'], 'content': content})
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        before, after = text.split(marker)
        return [self.tokenizer.encode(part, add_special_tokens=False) for part in (before, after)]

    def prepare(self, scene, question, answer=None, history=()):
        prefix, suffix = self.prompt_parts(question, history)
        answer_ids = [] if answer is None else self.tokenizer.encode(answer, add_special_tokens=False) + [self.stop_id]
        if len(answer_ids) > self.settings.max_answer_tokens:
            raise ValueError('Answer exceeds --max-answer-tokens; increase it explicitly (no silent truncation)')
        scene_embeds, coords = self.scene_tokens(scene)
        prompt_len = len(prefix) + len(scene_embeds) + len(suffix)
        length = prompt_len + len(answer_ids)
        if length > self.max_length:
            raise ValueError(f'{length:,} tokens exceed configured context {self.max_length:,}; no tokens were dropped')
        embed = self.decoder.embed_tokens
        prefix_ids = torch.tensor(prefix, device=self.device, dtype=torch.long)
        tail_ids = torch.tensor(suffix + answer_ids, device=self.device, dtype=torch.long)
        embeddings = torch.cat([embed(prefix_ids), scene_embeds, embed(tail_ids)], dim=0).unsqueeze(0)
        # Use the complete ROI bounds for all ablations, so text rotary positions are matched.
        tail_start = max(len(prefix), math.ceil(float(np.max(scene['hi'])) / self.settings.rope_metres_per_unit) + 1)
        prefix_positions = torch.arange(len(prefix), device=self.device, dtype=torch.float32).expand(3, -1)
        tail_positions = torch.arange(tail_start, tail_start + len(tail_ids), device=self.device,
                                      dtype=torch.float32).expand(3, -1)
        spatial_positions = torch.cat([prefix_positions, coords.T, tail_positions], dim=1)
        causal_positions = torch.arange(length, device=self.device, dtype=torch.float32)[None]
        positions = torch.cat([causal_positions, spatial_positions], dim=0).unsqueeze(1)
        self.last_sequence_length = length
        return embeddings, positions, prompt_len, answer_ids, tail_start + len(suffix)

    def forward(self, scene, question, answer, history=()):
        embeddings, positions, prompt_len, targets, _ = self.prepare(scene, question, answer, history)
        # Our single example is unpadded. Supplying this mask stops Transformers 4.57.1
        # from inferring packed sequences from position_ids and allocating a dense mask.
        attention_mask = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
        hidden = self.decoder(inputs_embeds=embeddings, position_ids=positions,
                              attention_mask=attention_mask, use_cache=False).last_hidden_state
        # Project only answer-predicting positions: avoid an L_scene x vocabulary allocation.
        selected = hidden[:, prompt_len - 1:prompt_len + len(targets) - 1]
        logits = self.lm_head(selected).float()
        labels = torch.tensor(targets, device=self.device, dtype=torch.long)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels)

    @torch.inference_mode()
    def answer(self, scene, question, max_new_tokens=256, history=()):
        self.eval()
        embeds, positions, _, _, next_text_position = self.prepare(scene, question, history=history)
        if embeds.shape[1] + max_new_tokens > self.max_length:
            raise ValueError('Requested generation exceeds context; no scene tokens were removed')
        out = self.decoder(inputs_embeds=embeds, position_ids=positions, use_cache=True)
        cache = out.past_key_values
        length = embeds.shape[1]
        generated = []
        for step in range(max_new_tokens):
            token = int(self.lm_head(out.last_hidden_state[:, -1]).argmax(-1).item())
            if token in self.stop_ids:
                break
            generated.append(token)
            if step == max_new_tokens - 1:
                break
            next_embed = self.decoder.embed_tokens(torch.tensor([[token]], device=self.device))
            p = float(next_text_position + step)
            pos = torch.tensor([float(length), p, p, p], device=self.device).reshape(4, 1, 1)
            out = self.decoder(inputs_embeds=next_embed, position_ids=pos, past_key_values=cache,
                               cache_position=torch.tensor([length], device=self.device), use_cache=True)
            cache, length = out.past_key_values, length + 1
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def adapter_state(self):
        return {name: value.detach().cpu() for name, value in self.named_parameters() if value.requires_grad}

    def save(self, directory, training=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'model_config.json', asdict(self.settings))
        temporary = directory / 'adapter.tmp.pt'
        torch.save(self.adapter_state(), temporary)
        temporary.replace(directory / 'adapter.pt')
        if training is not None:
            torch.save(training, directory / 'training.tmp.pt')
            (directory / 'training.tmp.pt').replace(directory / 'training.pt')

    def load_adapter(self, directory):
        state = torch.load(Path(directory) / 'adapter.pt', map_location='cpu', weights_only=True)
        required = {name for name, value in self.named_parameters() if value.requires_grad}
        if set(state) != required:
            raise ValueError('Adapter parameters do not match this model configuration')
        self.load_state_dict(state, strict=False)
