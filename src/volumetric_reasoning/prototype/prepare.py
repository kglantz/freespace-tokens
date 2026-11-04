"""Download/import ViCA QA, register an existing ROI, or build scene geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np

from .common import relative_entry, sha256, write_json, write_jsonl


def conversation_turns(conversation, source):
    if not isinstance(conversation, list) or len(conversation) < 2 or len(conversation) % 2:
        raise ValueError(f'Expected complete alternating question/answer turns in {source}')
    turns = []
    for i, turn in enumerate(conversation):
        role, aliases = ('assistant', ('gpt', 'assistant')) if i % 2 else ('user', ('human', 'user'))
        if not isinstance(turn, dict) or turn.get('from') not in aliases or not isinstance(turn.get('value'), str):
            raise ValueError(f'Invalid {role} turn in {source}:{i}')
        content = re.sub(r'<(?:image|video)>', '', turn['value']).strip() if role == 'user' else turn['value'].strip()
        if not content:
            raise ValueError(f'Empty {role} turn in {source}:{i}')
        turns.append({'role': role, 'content': content})
    return turns


def import_vica(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('QA output must be new or empty')
    paths = sorted(args.input.rglob('*.json')) if args.input.is_dir() else [args.input]
    root = args.input if args.input.is_dir() else args.input.parent
    rows, skipped = [], {}
    scene_allow = set(args.scenes.read_text().split()) if args.scenes else None
    raw_available = {}
    for path in paths:
        records = json.loads(path.read_text())
        if not isinstance(records, list):
            raise ValueError(f'Expected a list of ViCA records: {path}')
        task = path.stem
        if task == 'obj_appearance_order':
            skipped['appearance_order_requires_video'] = skipped.get('appearance_order_requires_video', 0) + len(records)
            continue
        for number, record in enumerate(records):
            video = record.get('video', '')
            video_path = PurePosixPath(video.replace('\\', '/')) if isinstance(video, str) else PurePosixPath()
            if 'scannetpp' not in video_path.parts[:-1] or video_path.stem in ('', '.', '..'):
                skipped['other_dataset'] = skipped.get('other_dataset', 0) + 1
                continue
            scene = video_path.stem
            if scene_allow is not None and scene not in scene_allow:
                skipped['scene_filter'] = skipped.get('scene_filter', 0) + 1
                continue
            if args.data_root:
                if scene not in raw_available:
                    iphone = args.data_root / 'data' / scene / 'iphone'
                    raw_available[scene] = all((iphone / name).is_file() for name in
                                               ('rgb.mkv', 'depth.bin', 'pose_intrinsic_imu.json'))
                if not raw_available[scene]:
                    skipped['raw_scene_not_available'] = skipped.get('raw_scene_not_available', 0) + 1
                    continue
            turns = conversation_turns(record.get('conversations'), f'{path}:{number}')
            key = f'{path.relative_to(root).with_suffix("").as_posix()}/{number}'
            for i in range(0, len(turns), 2):
                row = {'id': key if i == 0 else f'{key}/turn{i // 2 + 1}',
                       'scene_id': scene, 'region_id': scene, 'scope': 'scene',
                       'question': turns[i]['content'], 'answer': turns[i + 1]['content'], 'task': task,
                       'source_video': video, 'source_file': str(path)}
                if i:
                    row['history'] = turns[:i]
                rows.append(row)
    scenes = sorted({r['scene_id'] for r in rows})
    if not scenes:
        raise ValueError('No matching ScanNet++ QA; check raw data root and scene filter')
    if args.single_scene_debug:
        split = {'train': scenes, 'val': [], 'test': []}
    elif args.split_json:
        split = json.loads(args.split_json.read_text())
        sets = [set(split[k]) for k in ('train', 'val', 'test')]
        if (any(not s or len(s) != len(split[name]) for s, name in zip(sets, ('train', 'val', 'test')))
                or any(sets[i] & sets[j] for i in range(3) for j in range(i))
                or set.union(*sets) != set(scenes)):
            raise ValueError('Explicit train/val/test splits must be nonempty and cover available scenes exactly once')
    else:
        if len(scenes) < 3:
            raise ValueError('Need at least three scenes for disjoint splits; --single-scene-debug creates training-only data')
        if not (0 < args.val_fraction < 1 and 0 < args.test_fraction < 1 and args.val_fraction + args.test_fraction < 1):
            raise ValueError('Invalid split fractions')
        shuffled = np.random.default_rng(args.seed).permutation(scenes).tolist()
        nv, nt = max(1, round(len(scenes) * args.val_fraction)), max(1, round(len(scenes) * args.test_fraction))
        if nv + nt >= len(scenes):
            raise ValueError('Split fractions leave no training scenes')
        split = {'val': shuffled[:nv], 'test': shuffled[nv:nv + nt], 'train': shuffled[nv + nt:]}
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in ('train', 'val', 'test'):
        selected = [r for r in rows if r['scene_id'] in set(split[name])]
        write_jsonl(args.output / f'{name}.jsonl', selected)
        counts[name] = len(selected)
    write_json(args.output / 'splits.json', split)
    write_json(args.output / 'manifest.json', {'seed': args.seed, 'qa_counts': counts,
               'scene_counts': {k: len(v) for k, v in split.items()}, 'skipped': skipped,
               'training_only_debug': args.single_scene_debug,
               'source_sha256': {str(p): sha256(p) for p in paths},
               'scope': 'Original scene-wide annotations; do not silently pair these with cropped ROIs.'})
    print(json.dumps({'scenes': len(scenes), 'qa_counts': counts, 'skipped': skipped}), flush=True)


def register(args):
    if args.index.exists():
        data = json.loads(args.index.read_text())
    else:
        data = {'schema_version': 1, 'regions': {}}
    key = args.region_id or args.scene
    if key in data['regions']:
        raise ValueError(f'Region already registered: {key}')
    features = args.features or args.index.resolve().parent / key / 'surface_features.npz'
    data['regions'][key] = relative_entry(args.index, args.scene, args.volume, args.regions, features, args.scope)
    write_json(args.index, data)


def observed_bounds(scene, frames, voxel_size, min_depth, max_depth, padding):
    from ..geometry.backprojection import depth_to_points
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for i in frames:
        points, _ = depth_to_points(scene.load_depth(i), scene.K_depth(i), scene.cam_to_world(i), min_depth, max_depth)
        if len(points):
            lo, hi = np.minimum(lo, points.min(axis=0)), np.maximum(hi, points.max(axis=0))
    if not np.isfinite(lo).all():
        raise ValueError('No valid depth points for scene bounds')
    return np.r_[np.floor((lo - padding) / voxel_size) * voxel_size,
                 np.ceil((hi + padding) / voxel_size) * voxel_size].tolist()


def build_scenes(args):
    from ..datasets.scannetpp import ScanNetPPIphoneScene
    from .. import state_map, region_tokens
    split = json.loads((args.qa_dir / 'splits.json').read_text())
    scenes = sorted(set(s for group in split.values() for s in group))
    if args.scenes:
        allowed = set(args.scenes.read_text().split())
        scenes = [s for s in scenes if s in allowed]
    if not scenes:
        raise ValueError('No scenes selected for geometry construction')
    if not np.isfinite(args.padding) or args.padding < 0 or not np.isfinite(args.voxel_size) or args.voxel_size <= 0:
        raise ValueError('Padding must be nonnegative and voxel size must be positive, both finite')
    output = args.output.resolve()
    index_path = output / 'index.json'
    data = json.loads(index_path.read_text()) if index_path.exists() else {'schema_version': 1, 'regions': {}}
    for scene_id in scenes:
        directory = output / scene_id
        volume = directory / 'map' / 'volume.npz'
        regions = directory / 'regions' / 'token_geometry.npz'
        scene = ScanNetPPIphoneScene(args.data_root, scene_id)
        frames = scene.select_uniform(args.frames)
        bounds = observed_bounds(scene, frames, args.voxel_size, args.min_depth, args.max_depth, args.padding)
        if not volume.exists():
            state_map.main(['--data-root', str(args.data_root), '--scene', scene_id, '--frames', str(args.frames),
                            '--bounds', *map(str, bounds), '--voxel-size', str(args.voxel_size),
                            '--truncation', str(args.truncation), '--free-margin', str(args.free_margin),
                            '--min-depth', str(args.min_depth), '--max-depth', str(args.max_depth),
                            '--max-voxels', str(args.max_voxels), '--skip-depth-check', '--output', str(volume.parent)])
        else:
            manifest = json.loads((volume.parent / 'manifest.json').read_text())
            with np.load(volume, allow_pickle=False) as saved:
                same = (np.array_equal(saved['frame_indices'], frames) and np.isclose(float(saved['voxel_size']), args.voxel_size)
                        and np.isclose(float(saved['trunc']), args.truncation) and np.isclose(float(saved['free_margin']), args.free_margin))
            settings = manifest['fusion_settings']
            if (manifest['status'] != 'complete' or manifest['scene'] != scene_id or not same
                    or not np.allclose(manifest['requested_bounds'], bounds)
                    or settings['min_depth'] != args.min_depth or settings['max_depth'] != args.max_depth):
                raise ValueError(f'Existing map settings differ/incomplete for {scene_id}; choose a new output directory')
        if not regions.exists():
            region_tokens.main(['--source-volume', str(volume), '--output', str(regions.parent)])
        else:
            manifest = json.loads((regions.parent / 'manifest.json').read_text())
            if (manifest['status'] != 'complete' or manifest['source_sha256'] != sha256(volume)
                    or manifest['output_sha256']['token_geometry.npz'] != sha256(regions)):
                raise ValueError(f'Existing regions are incomplete or do not match the map: {scene_id}')
        data['regions'][scene_id] = relative_entry(index_path, scene_id, volume, regions,
                                                   directory / 'surface_features.npz', 'scene')
        write_json(index_path, data)
        print(f'Geometry complete: {scene_id}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('download-qa')
    d.add_argument('--output', type=Path, required=True)
    d.add_argument('--revision', default='main')
    q = sub.add_parser('import-vica')
    q.add_argument('--input', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--data-root', type=Path, help='Restrict to locally available raw ScanNet++ scenes')
    q.add_argument('--scenes', type=Path, help='Optional newline-separated scene IDs')
    q.add_argument('--split-json', type=Path)
    q.add_argument('--seed', type=int, default=42)
    q.add_argument('--val-fraction', type=float, default=.1)
    q.add_argument('--test-fraction', type=float, default=.1)
    q.add_argument('--single-scene-debug', action='store_true')
    r = sub.add_parser('register')
    r.add_argument('--index', type=Path, required=True)
    r.add_argument('--scene', required=True)
    r.add_argument('--region-id')
    r.add_argument('--volume', type=Path, required=True)
    r.add_argument('--regions', type=Path, required=True)
    r.add_argument('--features', type=Path)
    r.add_argument('--scope', choices=['roi', 'scene'], default='roi')
    b = sub.add_parser('build-scenes')
    b.add_argument('--qa-dir', type=Path, required=True)
    b.add_argument('--data-root', type=Path, required=True)
    b.add_argument('--output', type=Path, required=True)
    b.add_argument('--scenes', type=Path)
    b.add_argument('--frames', type=int, default=32)
    b.add_argument('--voxel-size', type=float, default=.02)
    b.add_argument('--truncation', type=float, default=.08)
    b.add_argument('--free-margin', type=float, default=.02)
    b.add_argument('--min-depth', type=float, default=.1)
    b.add_argument('--max-depth', type=float, default=4.)
    b.add_argument('--padding', type=float, default=.1)
    b.add_argument('--max-voxels', type=int, default=50_000_000)
    args = p.parse_args()
    if args.command == 'download-qa':
        from huggingface_hub import snapshot_download
        print(snapshot_download(repo_id='nkkbr/ViCA-322K', repo_type='dataset', revision=args.revision,
                                allow_patterns=['scannetpp/**/*.json'], local_dir=args.output))
    elif args.command == 'import-vica':
        import_vica(args)
    elif args.command == 'register':
        register(args)
    else:
        build_scenes(args)


if __name__ == '__main__':
    main()
