from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


VARIANTS = ('surface', 'surface_free', 'surface_unknown', 'surface_both')


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_index(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported scene index schema')
    regions = data['regions']
    for item in regions.values():
        for field in ('volume', 'regions', 'features'):
            item[field] = str((path.parent / item[field]).resolve())
    return regions


def relative_entry(index, scene_id, volume, regions, features, scope='roi'):
    return {'scene_id': scene_id, 'scope': scope,
            **{k: os.path.relpath(Path(v).resolve(), Path(index).resolve().parent)
               for k, v in (('volume', volume), ('regions', regions), ('features', features))}}


def check_rows(rows, index):
    if not rows:
        raise ValueError('QA file is empty')
    seen = set()
    for row in rows:
        for field in ('id', 'scene_id', 'region_id', 'question', 'answer'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'QA record needs a nonempty {field}')
        if row['id'] in seen:
            raise ValueError(f'Duplicate QA id: {row["id"]}')
        seen.add(row['id'])
        entry = index.get(row['region_id'])
        if entry is None:
            raise ValueError(f'QA region is missing from the index: {row["region_id"]}')
        if entry['scene_id'] != row['scene_id']:
            raise ValueError('QA scene ID does not match region index')
        if entry['scope'] != row.get('scope', 'scene'):
            raise ValueError(f'QA scope must match the indexed geometry for {row["id"]}. '
                             'Use scene-wide QA with scene geometry and ROI QA with its exact crop.')
        history = row.get('history', [])
        if (not isinstance(history, list) or len(history) % 2
                or any(not isinstance(turn, dict)
                       or turn.get('role') != ('assistant' if i % 2 else 'user')
                       or not isinstance(turn.get('content'), str) or not turn['content'].strip()
                       for i, turn in enumerate(history))):
            raise ValueError(f'QA history must contain complete user/assistant turns: {row["id"]}')


def load_scene(entry):
    with np.load(entry['features'], allow_pickle=False) as f, np.load(entry['regions'], allow_pickle=False) as g:
        if f['schema_version'].item() != 1 or g['schema_version'].item() != 1:
            raise ValueError('Unsupported feature/region schema')
        if f['region_sha256'].item() != sha256(entry['regions']):
            raise ValueError('Feature cache belongs to different region geometry; extract it again')
        surface_indices = g['surface_index']
        if not np.array_equal(f['surface_index'], surface_indices):
            raise ValueError('Feature and geometry surface indices differ')
        features, observations = f['features'], f['observation_count']
        if (features.ndim != 2 or features.shape[0] != len(surface_indices) or not features.shape[1]
                or observations.shape != (len(surface_indices),) or (observations < 0).any()
                or not np.isfinite(features).all()):
            raise ValueError(f'Invalid surface feature values or shape: {entry["features"]}')
        origin = g['origin']
        return {'features': features, 'has_feature': observations > 0,
                'surface_center': g['surface_center'], 'region_center': g['region_center'],
                'region_extent': g['region_extent'], 'region_state': g['region_state'],
                'origin': origin, 'hi': origin + g['dims'] * float(g['voxel_size']),
                'feature_dim': features.shape[1], 'encoder': str(f['encoder'].item()),
                'encoder_revision': str(f['encoder_revision'].item())}
