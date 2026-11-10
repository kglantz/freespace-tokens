"""Checks of model.py against Qwen3-VL itself, plus an overfitting test (about a minute on a GPU).

  1. chat format == Qwen's chat template (with an image in place of the scene) for every training example
  2. scene rotary positions == Qwen's get_rope_index for an image laid out as the same grid
  3. our decoder call == the stock model on plain text
  4. at initialization: surface token == its Qwen feature, box token == its type embedding
  5. answer-only loss == cross-entropy over full logits; gradients only reach trainable parameters
  6. cached greedy decoding == uncached greedy decoding
  7. overfitting one example the untrained model gets wrong: loss -> ~0, generation reproduces the answer

Run from the repo root after build_scenes.py and extract_features.py:  python tests/test_model.py
"""
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volumetric_reasoning.model import PREFIX, SUFFIX, FreeSpaceVLM, scene_rope_positions  # noqa: E402
from volumetric_reasoning.qa_dataset import SceneQADataset, load_qa, prompt  # noqa: E402

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)
m = FreeSpaceVLM(MODEL).to(dev).eval()
tok = m.tokenizer
pad = tok.convert_tokens_to_ids("<|image_pad|>")

# 1. chat format
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: E402
proc = AutoProcessor.from_pretrained(MODEL)
rows = load_qa(ROOT / "data/qa/train.jsonl")
for r in rows:
    k = 7
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt(r)}]},
            {"role": "assistant", "content": [{"type": "text", "text": r["answer"]}]}]
    ref = proc.apply_chat_template(msgs, tokenize=False).replace("<|image_pad|>", "<|image_pad|>" * k)
    ours = (m._ids(PREFIX) + [pad] * k + m._ids(SUFFIX.format(prompt=prompt(r))) + m._ids(r["answer"])
            + [m.end_id] + m._ids("\n"))
    assert ours == tok(ref, add_special_tokens=False)["input_ids"], r["id"]
print(f"1. chat format identical to Qwen's template for all {len(rows)} training examples")

# 2. rotary positions
stock = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
H, W = 3, 5
rr, cc = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
xyz = torch.stack([cc.flatten() * 0.16, rr.flatten() * 0.16, torch.zeros(H * W)], 1) + torch.tensor([3.0, -2.0, 0.7])
s = len(m._ids(PREFIX))
text = m._ids(SUFFIX.format(prompt="Q?"))
ids = torch.tensor([m._ids(PREFIX) + [pad] * (H * W) + text])
ref_pos, _ = stock.model.get_rope_index(ids, (ids == pad).int(), image_grid_thw=torch.tensor([[1, 2 * H, 2 * W]]))
scene_pos, nxt = scene_rope_positions(xyz, s)
ours = torch.cat([torch.arange(s).float().expand(3, -1), scene_pos, (nxt + torch.arange(len(text)).float()).expand(3, -1)], 1)
assert torch.allclose(ours, ref_pos[:, 0].float().cpu(), atol=1e-4)
print("2. scene rotary positions == Qwen get_rope_index for the same grid (incl. following text)")

# 3. decoder call == stock model on text
ids = torch.tensor([tok("<|im_start|>user\nHow far is the sofa from the door?<|im_end|>\n<|im_start|>assistant\n")["input_ids"]], device=dev)
with torch.no_grad():
    ref = stock(input_ids=ids).logits[0].float()
    pos = torch.arange(ids.shape[1], device=dev).float().expand(3, 1, -1)
    ours = m.lm_head(m.decoder(inputs_embeds=m.decoder.embed_tokens(ids), position_ids=pos).last_hidden_state[0]).float()
assert (ours.argmax(-1) == ref.argmax(-1)).all() and (ours - ref).abs().max() < 0.1, (ours - ref).abs().max()
print(f"3. decoder call == stock model on text (max |logit diff| {(ours - ref).abs().max():.4f})")
del stock

# 4. initialization
ds = SceneQADataset(ROOT / "data/qa/train.jsonl", "surface_both", ROOT / "outputs/scenes", MODEL, scenes={"81a82c3618"})
ex = ds[0]
exd = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in ex.items()}
with torch.no_grad():
    z = m.scene_encoder(exd)
ns = len(ex["surface_feature"])
assert torch.equal(z[:ns], exd["surface_feature"].float())
assert torch.equal(z[ns:], m.scene_encoder.kind.weight[torch.where(exd["box_type"] == 1, 1, 2)])
print(f"4. init: {ns} surface tokens == Qwen features, {len(z) - ns} boxes == type embeddings")

# 5. loss and gradients
x, pos, n = m.inputs(ex, answer=True)
target = torch.tensor(m._ids(ex["answer"]) + [m.end_id], device=dev)
with torch.no_grad():
    full = m.lm_head(m.decoder(inputs_embeds=x, position_ids=pos).last_hidden_state[0]).float()
    assert abs(m.loss(ex).item() - F.cross_entropy(full[-n - 1:-1], target).item()) < 1e-3
m.train()
m.loss(ex).backward()
no_grad = [k for k, p in m.named_parameters() if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
assert not [k for k, p in m.named_parameters() if not p.requires_grad and p.grad is not None]
assert all(("lora_" in k) or k.startswith("scene_encoder.") for k, p in m.named_parameters() if p.requires_grad)
# zero-initialized output layers: their branch's first layer (and LoRA A) gets no gradient at step 0
first = {f"scene_encoder.{b}.{w}" for b in ("adapter.0", "position.mlp.0", "size.mlp.0") for w in ("weight", "bias")}
assert all("lora_A" in k or k in first for k in no_grad), no_grad
m.zero_grad(set_to_none=True)
print(f"5. loss == full-logit cross-entropy; gradients only on the "
      f"{sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6:.1f} M trainable parameters")

# 6. cached == uncached decoding (LoRA B perturbed, so the model differs from the base model)
m.eval()
with torch.no_grad():
    saved = {k: p.clone() for k, p in m.named_parameters() if "lora_B" in k}
    for k, p in m.named_parameters():
        if "lora_B" in k:
            p.normal_(std=1e-3)
    cached = m.generate(ex, max_new_tokens=8)
    x, pos, _ = m.inputs(ex, answer=False)
    toks = []
    for _ in range(8):
        t = int(m.lm_head(m.decoder(inputs_embeds=x, position_ids=pos).last_hidden_state[0, -1]).argmax())
        if t == m.end_id:
            break
        toks.append(t)
        x = torch.cat([x, m.decoder.embed_tokens(torch.tensor([[t]], device=dev))], 1)
        pos = torch.cat([pos, pos[:, :, -1:] + 1], 2)
    assert cached == tok.decode(toks, skip_special_tokens=True), (cached, tok.decode(toks))
    for k, p in m.named_parameters():
        if "lora_B" in k:
            p.copy_(saved[k])
print(f"6. cached decoding == uncached decoding ({cached!r})")

# 7. overfit the distance question of 81a82c3618 that the untrained model answers wrongly (1.0 instead of 1.3)
ex = next(ds[i] for i in range(len(ds)) if ds.rows[i]["task"] == "object_abs_distance")
before = m.generate(ex)
m.train()
params = [p for p in m.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.0)
for step in range(12):
    opt.zero_grad(set_to_none=True)
    loss = m.loss(ex)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
m.eval()
after = m.generate(ex)
print(f"7. overfitting: answer {ex['answer']!r}, before {before!r}, after {after!r}, final loss {loss.item():.4f}")
assert after == ex["answer"] and loss.item() < 0.01
print("All model checks passed.")
