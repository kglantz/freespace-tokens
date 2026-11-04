"""Compare predictions from the four ablations on identical QA."""
import argparse
import json
from pathlib import Path

from .common import VARIANTS, read_jsonl, write_json
from .metrics import summarize


def compare(directory):
    result, reference, provenances = {}, None, []
    for variant in VARIANTS:
        path = directory / f'{variant}.jsonl'
        rows = read_jsonl(path)
        if not rows or len({r['id'] for r in rows}) != len(rows):
            raise ValueError(f'Predictions must contain nonempty, unique QA records: {path}')
        identity = [
            (r['id'], r['scene_id'], r['region_id'], r.get('scope'), r.get('task'),
             r.get('history', []), r['question'], r['answer'])
            for r in rows
        ]
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError('Ablations must be evaluated on exactly the same ordered QA examples')
        if any(r['variant'] != variant for r in rows):
            raise ValueError(f'Prediction file has wrong ablation: {variant}')
        metrics_path = path.with_suffix('.metrics.json')
        provenance = json.loads(metrics_path.read_text()).get('provenance') if metrics_path.is_file() else None
        # Checkpoint selection can pick different steps under the same training schedule.
        if provenance is not None:
            result_step = provenance.pop('selected_step', None)
        else:
            result_step = None
        provenances.append(provenance)
        result[variant] = summarize(rows)
        if result_step is not None:
            result[variant]['selected_step'] = result_step
        lengths = [r.get('input_tokens') for r in rows]
        if all(isinstance(length, int) and length > 0 for length in lengths):
            result[variant]['input_tokens'] = {'mean': sum(lengths) / len(lengths), 'max': max(lengths)}
    if any(p is not None for p in provenances) and any(p != provenances[0] for p in provenances):
        raise ValueError('Ablation provenance differs (data, model, seed, or inference settings)')
    required = ('source_identity', 'seed', 'training_options', 'world_size', 'model',
                'train_scenes', 'val_scenes', 'eval_feature_sha256', 'max_new_tokens')
    verified = provenances[0] is not None and all(provenances[0].get(k) is not None for k in required)
    for metrics in result.values():
        metrics['matched_provenance_verified'] = verified
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.suffix != '.json':
        p.error('--output must be a .json file (a Markdown report is written beside it)')
    result = compare(args.predictions_dir)
    write_json(args.output, result)
    lines = ['# Prototype ablation comparison', '', 'Diagnostic scores on the supplied QA.', '']
    if not next(iter(result.values()))['matched_provenance_verified']:
        lines += ['Full training and inference provenance was unavailable. QA identities were matched.', '']
    lines += ['| Variant | Examples | Normalized exact match | Numeric accuracy at 10% | Mean input tokens |',
              '|---|---:|---:|---:|---:|']
    for variant, metrics in result.items():
        overall = metrics['groups']['all']
        n = overall.get('numeric_relative_accuracy_0.1')
        mean_tokens = metrics.get('input_tokens', {}).get('mean')
        numeric_label = f'{n:.4f}' if n is not None else 'n/a'
        token_label = f'{mean_tokens:.1f}' if mean_tokens is not None else 'n/a'
        lines.append(f'| {variant} | {overall["examples"]} | {overall["normalized_exact_match"]:.4f} | {numeric_label} | {token_label} |')
    args.output.with_suffix('.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
