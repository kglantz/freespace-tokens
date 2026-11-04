"""Greedy cached generation for one question or a held-out QA file."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from .common import check_rows, load_index, read_jsonl, sha256, write_json, write_jsonl
from .metrics import summarize
from .model import ModelConfig, SpatialVLM
from .train import SceneCache, check_scene_context, count_scene_tokens


def checkpoint_provenance(checkpoint):
    manifest_path = checkpoint / 'data_manifest.json'
    run_path = checkpoint.parent / 'run.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    run = json.loads(run_path.read_text()) if run_path.exists() else {}
    if 'adapter_sha256' in manifest and sha256(checkpoint / 'adapter.pt') != manifest['adapter_sha256']:
        raise ValueError('Adapter does not match checkpoint provenance. Finish or resume saving before evaluation.')
    if manifest and (manifest.get('source_identity') != run.get('source_identity')
                     or manifest.get('train_scenes') != run.get('train_scenes')):
        run = {}
    return {
        'train_scenes': manifest.get('train_scenes', run.get('train_scenes')),
        'val_scenes': manifest.get('val_scenes', run.get('val_scenes')),
        'source_identity': manifest.get('source_identity', run.get('source_identity')),
        'seed': manifest.get('seed', run.get('arguments', {}).get('seed')),
        'training_options': manifest.get('training_options'),
        'world_size': manifest.get('world_size', run.get('world_size')),
        'selected_step': manifest.get('global_step'),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--region-id')
    p.add_argument('--question')
    p.add_argument('--qa', type=Path, help='Generate all questions in this JSONL, with answer metrics')
    p.add_argument('--output', type=Path, help='Prediction JSONL; required with --qa')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--device', default='cuda')
    p.add_argument('--attention', choices=['sdpa', 'flash_attention_2'])
    p.add_argument('--limit', type=int, default=0, help='0 = every QA example')
    p.add_argument('--allow-seen-scenes', '--allow-training-scenes', action='store_true',
                   help='Debug evaluation on training or validation scenes, or without split provenance')
    args = p.parse_args()
    if args.max_new_tokens < 1 or args.limit < 0:
        raise ValueError('Generation length must be positive and limit nonnegative')
    if args.qa and (args.question or args.region_id or not args.output):
        raise ValueError('Use either --qa --output, or --region-id --question')
    if not args.qa and not (args.question and args.region_id):
        raise ValueError('Provide --qa or both --region-id and --question')
    if args.output and args.output.exists():
        raise ValueError('Prediction output already exists; choose a new path')
    index = load_index(args.index)
    if args.qa:
        rows = read_jsonl(args.qa)
        check_rows(rows, index)
        provenance = checkpoint_provenance(args.checkpoint)
        if not args.allow_seen_scenes:
            trained, validated = provenance['train_scenes'], provenance['val_scenes']
            if trained is None or validated is None:
                raise ValueError('Checkpoint lacks training or validation scene provenance. '
                                 'Use --allow-seen-scenes only for debug evaluation.')
            overlap = (set(trained) | set(validated)) & {r['scene_id'] for r in rows}
            if overlap:
                raise ValueError(f'Evaluation includes training or validation scenes: {sorted(overlap)}')
        if args.limit:
            rows = rows[:args.limit]
        provenance['eval_feature_sha256'] = {
            key: sha256(index[key]['features']) for key in sorted({r['region_id'] for r in rows})
        }
        provenance['max_new_tokens'] = args.max_new_tokens
        provenance['allow_seen_scenes'] = args.allow_seen_scenes
    else:
        rows = [{'id': 'interactive', 'region_id': args.region_id, 'question': args.question}]
    config = ModelConfig(**json.loads((args.checkpoint / 'model_config.json').read_text()))
    if args.attention:
        config.attention = args.attention
    if args.max_new_tokens >= config.max_sequence_tokens:
        raise ValueError('Requested generation leaves no context for the scene and prompt')
    counts = count_scene_tokens(index, sorted({r['region_id'] for r in rows}), config.variant)
    check_scene_context(counts, config.max_sequence_tokens - args.max_new_tokens)
    model = SpatialVLM(config, args.device, gradient_checkpointing=False)
    model.load_adapter(args.checkpoint)
    check_scene_context(counts, model.max_length - args.max_new_tokens)
    model.eval()
    cache = SceneCache(index)
    device = torch.device(args.device)
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[config.dtype]
    predictions = []
    for row in rows:
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == 'cuda' and dtype != torch.float32):
            prediction = model.answer(cache[row['region_id']], row['question'], args.max_new_tokens,
                                      history=row.get('history', ()))
        result = {**row, 'prediction': prediction, 'variant': config.variant,
                  'checkpoint': str(args.checkpoint), 'input_tokens': model.last_sequence_length}
        predictions.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent) as temp:
            staged = Path(temp)
            write_jsonl(staged / 'predictions.jsonl', predictions)
            if args.qa:
                provenance['model'] = {k: v for k, v in asdict(config).items() if k != 'variant'}
                write_json(staged / 'metrics.json', {**summarize(predictions), 'provenance': provenance})
            # Publish a complete file without overwriting another invocation's results.
            os.link(staged / 'predictions.jsonl', args.output)
            if args.qa:
                (staged / 'metrics.json').replace(args.output.with_suffix('.metrics.json'))


if __name__ == '__main__':
    main()
