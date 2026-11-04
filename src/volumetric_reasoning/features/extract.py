"""Cache frozen DINOv2 features, backproject RGB-D, and max-pool by surface cell.

Full images are resized without cropping, to patch-aligned dimensions. Valid
depth pixels select the corresponding image patch. Endpoints inside the grid use
their containing occupied cell or their nearest surface center within a fixed
metric radius. Unmatched surface cells retain zero visual features and an
explicit observation count.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch
from transformers import AutoImageProcessor, AutoModel

from ..datasets.scannetpp import ScanNetPPIphoneScene
from ..geometry.backprojection import depth_to_points
from ..geometry.tsdf import STATE_DEFINITION, STATE_SCHEMA_VERSION
from ..prototype.common import load_index, sha256, write_json


class FrozenEncoder:
    def __init__(self, name, revision, device, image_long_edge):
        self.name, self.device = name, torch.device(device)
        self.model = AutoModel.from_pretrained(name, revision=revision).to(self.device).eval()
        self.model.requires_grad_(False)
        if self.model.config.model_type != 'dinov2':
            raise ValueError('This extractor expects DINOv2 without register tokens')
        self.patch = int(self.model.config.patch_size)
        self.dim = int(self.model.config.hidden_size)
        self.revision = self.model.config._commit_hash or revision
        self.processor = AutoImageProcessor.from_pretrained(name, revision=self.revision, use_fast=False)
        self.long_edge = image_long_edge
        self.compute_dtype = torch.float32
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                if torch.cuda.is_bf16_supported(including_emulation=False):
                    self.compute_dtype = torch.bfloat16

    @torch.inference_mode()
    def encode(self, rgb):
        height, width = rgb.shape[:2]
        scale = self.long_edge / max(height, width)
        w = max(self.patch, round(width * scale / self.patch) * self.patch)
        h = max(self.patch, round(height * scale / self.patch) * self.patch)
        image = Image.fromarray(rgb).resize((w, h), Image.Resampling.BICUBIC)
        pixels = self.processor(images=image, do_resize=False, do_center_crop=False,
                                return_tensors='pt')['pixel_values'].to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.compute_dtype == torch.bfloat16):
            features = self.model(pixel_values=pixels).last_hidden_state[0, 1:].float()
        gh, gw = h // self.patch, w // self.patch
        if len(features) != gh * gw:
            raise RuntimeError('Encoder patch count does not match the resized image')
        return features, gh, gw


def _validated_inputs(entry):
    volume, geometry = Path(entry['volume']), Path(entry['regions'])
    volume_hash, geometry_hash = sha256(volume), sha256(geometry)
    state = json.loads((volume.parent / 'manifest.json').read_text())
    regions = json.loads((geometry.parent / 'manifest.json').read_text())
    if state.get('status') != 'complete' or state.get('scene') != entry['scene_id']:
        raise ValueError('State map is incomplete or belongs to a different scene')
    if (regions.get('status') != 'complete' or regions.get('source_sha256') != volume_hash
            or regions.get('output_sha256', {}).get('token_geometry.npz') != geometry_hash):
        raise ValueError('Region geometry is incomplete or does not match the registered state map')
    with np.load(volume, allow_pickle=False) as a:
        frames = a['frame_indices']
        source_h = float(a['voxel_size'])
        if (a['schema_version'].item() != STATE_SCHEMA_VERSION or a['state_definition'].item() != STATE_DEFINITION
                or frames.ndim != 1 or frames.dtype.kind not in 'iu' or not len(frames)
                or frames[0] < 0 or np.any(frames[1:] <= frames[:-1])
                or not np.array_equal(frames, state['frame_indices'])):
            raise ValueError('Invalid state map schema or frame provenance')
    with np.load(geometry, allow_pickle=False) as a:
        indices, centers = a['surface_index'], a['surface_center']
        origin, h, dims = a['origin'], float(a['voxel_size']), a['dims']
        if (a['schema_version'].item() != 1 or a['state_definition'].item() != STATE_DEFINITION
                or h != source_h or not np.array_equal(frames, a['frame_indices'])):
            raise ValueError('Map and regions must use identical observations and voxel size')
    return (volume_hash, geometry_hash, frames.tolist(), indices, centers, origin, h, dims,
            state['fusion_settings'])


def extract(entry, data_root, encoder, radius=.05, skip_existing=False):
    output = Path(entry['features'])
    if output.suffix != '.npz':
        raise ValueError('Feature cache path must end in .npz')
    (volume_hash, geometry_hash, frames, indices, centers, origin, h, dims, fusion) = _validated_inputs(entry)
    min_depth, max_depth = fusion['min_depth'], fusion['max_depth']
    calibration = Path(data_root) / 'data' / entry['scene_id'] / 'iphone' / 'pose_intrinsic_imu.json'
    signature = {'volume_sha256': volume_hash, 'region_sha256': geometry_hash,
                 'encoder': encoder.name, 'encoder_revision': encoder.revision,
                 'image_long_edge': encoder.long_edge, 'association_radius_m': radius,
                 'depth_range_m': [min_depth, max_depth], 'calibration_sha256': sha256(calibration),
                 'compute_dtype': str(encoder.compute_dtype), 'association_policy': 'in_grid_endpoints_v1'}
    metadata = output.with_suffix('.json')
    if output.exists():
        if not skip_existing:
            raise FileExistsError(f'Feature cache already exists: {output}')
        with np.load(output, allow_pickle=False) as cached:
            saved = json.loads(cached['metadata_json'].item()) if 'metadata_json' in cached else {}
        checksum = sha256(output)
        if (any(saved.get(k) != v for k, v in signature.items())
                or (metadata.exists() and json.loads(metadata.read_text()).get('cache_sha256') != checksum)):
            raise ValueError(f'Existing feature cache settings/content differ: {output}')
        if not metadata.exists():
            write_json(metadata, {**saved, 'cache_sha256': checksum})
        print(f'Reusing {output}', flush=True)
        return
    scene = ScanNetPPIphoneScene(data_root, entry['scene_id'])
    occupied_keys = np.ravel_multi_index(indices.T, dims)
    if np.any(np.diff(occupied_keys) <= 0):
        raise ValueError('Surface indices must be sorted and unique')
    tree = cKDTree(centers) if len(centers) else None
    pooled = torch.full((len(indices), encoder.dim), -torch.inf, dtype=torch.float32, device=encoder.device)
    counts = np.zeros(len(indices), np.int64)
    valid_points = inside_points = direct_points = accepted_points = 0
    for number, (frame_id, rgb) in enumerate(scene.iter_rgb(frames), 1):
        feature, gh, gw = encoder.encode(rgb)
        depth = scene.load_depth(frame_id)
        points, mask = depth_to_points(depth, scene.K_depth(frame_id), scene.cam_to_world(frame_id), min_depth, max_depth)
        rows, cols = np.nonzero(mask)
        valid_points += len(points)
        # Nearest patch assignment follows pixel-corner resize geometry exactly.
        patch_row = np.minimum(((rows + .5) / depth.shape[0] * gh).astype(int), gh - 1)
        patch_col = np.minimum(((cols + .5) / depth.shape[1] * gw).astype(int), gw - 1)
        cell = np.floor((points - origin) / h).astype(np.int64)
        inside = ((cell >= 0) & (cell < dims)).all(axis=1)
        inside_ids = np.flatnonzero(inside)
        inside_points += len(inside_ids)
        destination = np.full(len(points), -1, np.int64)
        if len(occupied_keys) and len(inside_ids):
            query_keys = np.ravel_multi_index(cell[inside].T, dims)
            match_ids = np.searchsorted(occupied_keys, query_keys)
            found = match_ids < len(occupied_keys)
            found[found] = occupied_keys[match_ids[found]] == query_keys[found]
            destination[inside_ids[found]] = match_ids[found]
            direct_points += int(found.sum())
        unmatched = np.flatnonzero(inside & (destination < 0))
        if tree is not None and len(unmatched):
            distance, nearest = tree.query(points[unmatched], distance_upper_bound=radius)
            supported = np.isfinite(distance)
            destination[unmatched[supported]] = nearest[supported]
        keep = destination >= 0
        ids = destination[keep]
        if len(ids):
            patch_ids = torch.as_tensor((patch_row * gw + patch_col)[keep], device=encoder.device)
            targets = torch.as_tensor(ids, device=encoder.device)
            for start in range(0, len(ids), 4096):
                block = slice(start, start + 4096)
                pooled.scatter_reduce_(0, targets[block, None].expand(-1, encoder.dim), feature[patch_ids[block]],
                                       reduce='amax', include_self=True)
            counts += np.bincount(ids, minlength=len(counts))
            accepted_points += len(ids)
        print(f'{entry["scene_id"]}: features {number}/{len(frames)}', flush=True)
    pooled[torch.as_tensor(counts == 0, device=encoder.device)] = 0
    features = pooled.cpu().numpy().astype(np.float16)
    if not np.isfinite(features).all():
        raise FloatingPointError(f'Non-finite pooled features in scene {entry["scene_id"]}')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        **signature, 'scene_id': entry['scene_id'],
        'surface_tokens': len(indices), 'observed_surface_tokens': int((counts > 0).sum()),
        'zero_feature_surface_tokens': int((counts == 0).sum()), 'feature_dim': encoder.dim,
        'valid_depth_points': valid_points, 'in_grid_depth_points': inside_points,
        'assigned_depth_points': accepted_points, 'direct_depth_points': direct_points,
        'nearest_depth_points': accepted_points - direct_points, 'frame_indices': frames,
        'association': 'in-grid depth endpoints use their occupied cell or nearest occupied center within radius',
        'missing_feature_policy': 'retain every surface cell; zero vector when no accepted image observation',
        'pooling': 'channel-wise maximum across all assigned depth-pixel features and views',
    }
    with TemporaryDirectory(dir=output.parent) as temp:
        temporary = Path(temp) / 'features.npz'
        np.savez_compressed(temporary, schema_version=1, **signature, frame_indices=frames,
                            surface_index=indices, surface_center=centers, observation_count=counts,
                            features=features, metadata_json=json.dumps(report))
        report['cache_sha256'] = sha256(temporary)
        write_json(Path(temp) / 'features.json', report)
        os.link(temporary, output)
        (Path(temp) / 'features.json').replace(metadata)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--encoder', default='facebook/dinov2-base')
    p.add_argument('--revision', default='main')
    p.add_argument('--image-long-edge', type=int, default=1008)
    p.add_argument('--association-radius', type=float, default=.05)
    p.add_argument('--device', default='cuda')
    p.add_argument('--skip-existing', action='store_true')
    p.add_argument('--region-id', help='Process just this region')
    args = p.parse_args()
    if args.image_long_edge < 14 or not np.isfinite(args.association_radius) or args.association_radius <= 0:
        raise ValueError('Image size and association radius must be positive')
    local_rank, rank, world = (int(os.getenv(k, default)) for k, default in
                               (('LOCAL_RANK', '0'), ('RANK', '0'), ('WORLD_SIZE', '1')))
    device = f'cuda:{local_rank}' if args.device == 'cuda' else args.device
    if str(device).startswith('cuda'):
        torch.cuda.set_device(device)
    encoder = FrozenEncoder(args.encoder, args.revision, device, args.image_long_edge)
    index = load_index(args.index)
    keys = [args.region_id] if args.region_id else sorted(index)
    for key in keys[rank::world]:
        extract(index[key], args.data_root, encoder, args.association_radius, args.skip_existing)


if __name__ == '__main__':
    main()
