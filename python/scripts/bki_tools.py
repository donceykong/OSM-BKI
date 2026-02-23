#!/usr/bin/env python3
"""
BKI map tools: convert and visualize .bki map files.

Subcommands:
  convert   - Export .bki to point cloud (.bin) and labels (.label)
  visualize - Display .bki voxels in an Open3D viewer, colored by semantic label

The .bki format (version 2/3) stores a sparse block grid. Both commands read it
via the same read_bki() logic, filtering voxels by min alpha sum.
"""

import argparse
import struct
import sys
import yaml
import numpy as np
import open3d as o3d
from pathlib import Path

from osm_loader import OSMLoader

BLOCK_SIZE = 8

# Fallback label colors when config has no colors block
MCD_LABEL_COLORS = {
    0: [0.5, 0.5, 0.5],    # barrier
    1: [0.0, 0.0, 1.0],    # bike
    2: [0.8, 0.2, 0.2],    # building
    7: [0.6, 0.3, 0.1],    # fence
    13: [0.4, 0.4, 0.4],   # parkinglot
    14: [1.0, 0.5, 0.0],   # pedestrian
    15: [0.7, 0.7, 0.0],   # pole
    16: [0.2, 0.2, 0.2],   # road
    18: [0.7, 0.7, 0.7],   # sidewalk
    22: [0.0, 0.8, 0.8],   # traffic-sign
    24: [0.4, 0.3, 0.1],   # treetrunk
    25: [0.2, 0.8, 0.2],   # vegetation
    26: [0.0, 0.5, 1.0],   # vehicle-dynamic
}

KITTI_LABEL_COLORS = {
    0: [0.0, 0.0, 0.0],    # unlabeled
    10: [0.0, 0.0, 1.0],   # car
    30: [1.0, 0.5, 0.0],   # person
    40: [0.2, 0.2, 0.2],   # road
    44: [0.4, 0.4, 0.4],   # parking
    48: [0.7, 0.7, 0.7],   # sidewalk
    50: [0.8, 0.2, 0.2],   # building
    51: [0.6, 0.3, 0.1],   # fence
    70: [0.2, 0.8, 0.2],   # vegetation
    71: [0.4, 0.3, 0.1],   # trunk
    80: [0.7, 0.7, 0.0],   # pole
    81: [0.0, 0.8, 0.8],   # traffic-sign
}


def load_visualize_config(config_path):
    """
    Load config for visualization: colors (normalized to [0,1]) and OSM params.
    Returns dict with: colors_by_label, osm_path, osm_origin_lat, osm_origin_lon,
    osm_world_offset_x, osm_world_offset_y. OSM fields are None if not configured.
    """
    config_dir = Path(config_path).resolve().parent
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    out = {
        'colors_by_label': {},
        'osm_path': None,
        'osm_origin_lat': None,
        'osm_origin_lon': None,
        'osm_world_offset_x': 0.0,
        'osm_world_offset_y': 0.0,
    }

    if 'colors' in raw:
        for lbl_str, rgb in raw['colors'].items():
            out['colors_by_label'][int(lbl_str)] = [c / 255.0 for c in rgb]

    # OSM origin (used when --osm override is passed or when osm_file is in config)
    if raw.get('osm_origin_lat') is not None and raw.get('osm_origin_lon') is not None:
        out['osm_origin_lat'] = float(raw['osm_origin_lat'])
        out['osm_origin_lon'] = float(raw['osm_origin_lon'])
    elif 'init_latlon_day_06' in raw:
        ll = raw['init_latlon_day_06']
        out['osm_origin_lat'] = float(ll[0])
        out['osm_origin_lon'] = float(ll[1])
    out['osm_world_offset_x'] = float(raw.get('osm_world_offset_x', 0.0))
    out['osm_world_offset_y'] = float(raw.get('osm_world_offset_y', 0.0))

    if 'osm_file' in raw:
        root = raw.get('dataset_root_path')
        osm_file = raw['osm_file']
        if root:
            out['osm_path'] = str(Path(root) / osm_file)
        else:
            out['osm_path'] = str(config_dir / osm_file)
        if not Path(out['osm_path']).exists():
            out['osm_path'] = None

    return out


def read_bki(bki_path, config_path, min_alpha_sum=1e-6, min_max_prob=0.0):
    """
    Read .bki file and return (points_xyz, labels_raw) as arrays.
    points_xyz: (N, 3) float32, labels_raw: (N,) uint32.
    Filters: sum(alpha) >= min_alpha_sum, and max(alpha)/sum(alpha) >= min_max_prob.
    """
    with open(config_path) as f:
        data = yaml.safe_load(f)
    labels = data.get("labels") or {}
    raw_ids = sorted(int(k) for k in labels.keys())
    K = len(raw_ids)
    dense_to_raw = raw_ids

    block_alpha_size = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE * K

    points_list = []
    labels_list = []

    with open(bki_path, "rb") as f:
        version = struct.unpack("B", f.read(1))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported .bki version {version} (expected 2 or 3)")

        resolution = struct.unpack("f", f.read(4))[0]
        l_scale = struct.unpack("f", f.read(4))[0]
        sigma_0 = struct.unpack("f", f.read(4))[0]

        current_time = 0
        if version >= 3:
            current_time = struct.unpack("i", f.read(4))[0]

        num_blocks = struct.unpack("Q", f.read(8))[0]

        for _ in range(num_blocks):
            bx, by, bz = struct.unpack("iii", f.read(12))

            last_updated = 0
            if version >= 3:
                last_updated = struct.unpack("i", f.read(4))[0]

            alpha_bytes = f.read(block_alpha_size * 4)
            if len(alpha_bytes) != block_alpha_size * 4:
                break

            alpha = np.frombuffer(alpha_bytes, dtype=np.float32)
            alpha = alpha.reshape((BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, K))

            for lz in range(BLOCK_SIZE):
                for ly in range(BLOCK_SIZE):
                    for lx in range(BLOCK_SIZE):
                        a = alpha[lx, ly, lz, :]
                        s = float(np.sum(a))
                        if s < min_alpha_sum:
                            continue

                        if np.max(a) - np.min(a) < 1e-6:
                            continue

                        max_prob = np.max(a) / s if s > 0 else 0.0
                        if max_prob < min_max_prob:
                            continue

                        best = int(np.argmax(a))
                        raw = dense_to_raw[best] if best < len(dense_to_raw) else 0

                        vx = bx * BLOCK_SIZE + lx
                        vy = by * BLOCK_SIZE + ly
                        vz = bz * BLOCK_SIZE + lz
                        x = (vx + 0.5) * resolution
                        y = (vy + 0.5) * resolution
                        z = (vz + 0.5) * resolution

                        points_list.append([x, y, z])
                        labels_list.append(raw)

    if not points_list:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint32)

    points_xyz = np.array(points_list, dtype=np.float32)
    labels_raw = np.array(labels_list, dtype=np.uint32)
    return points_xyz, labels_raw


def get_label_colors(labels, label_format="auto", colors_from_config=None):
    """(N,) uint32 labels -> (N, 3) RGB. Uses config colors when provided."""
    colors = np.zeros((len(labels), 3))
    if colors_from_config:
        color_map = colors_from_config
    elif label_format == "auto":
        unique = np.unique(labels)
        if np.max(unique) > 30 or any(l in (40, 44, 48, 50, 70, 80, 81) for l in unique):
            color_map = KITTI_LABEL_COLORS
        else:
            color_map = MCD_LABEL_COLORS
    else:
        color_map = KITTI_LABEL_COLORS if label_format == "kitti" else MCD_LABEL_COLORS
    for i, label in enumerate(labels):
        label = int(label)
        if label in color_map:
            colors[i] = color_map[label]
        else:
            colors[i] = [1.0, 0.0, 1.0]
    return colors


def cmd_convert(args):
    """Export .bki to .bin and .label files."""
    bki_path = Path(args.bki)
    if not bki_path.exists():
        raise SystemExit(f"File not found: {bki_path}")
    if not Path(args.config).exists():
        raise SystemExit(f"Config not found: {args.config}")

    stem = bki_path.stem
    out_bin = args.output_bin or str(bki_path.parent / f"{stem}_map.bin")
    out_label = args.output_label or str(bki_path.parent / f"{stem}_map.label")

    print(f"Reading {bki_path} with config {args.config}...")
    points_xyz, labels_raw = read_bki(str(bki_path), args.config, min_alpha_sum=args.min_alpha, min_max_prob=args.min_max_prob)
    n = len(points_xyz)
    print(f"Exporting {n} voxels.")

    cloud = np.hstack([points_xyz, np.zeros((n, 1), dtype=np.float32)])
    cloud.tofile(out_bin)
    labels_raw.tofile(out_label)
    print(f"Wrote {out_bin}")
    print(f"Wrote {out_label}")
    return 0


def cmd_visualize(args):
    """Display .bki voxels in Open3D viewer, with optional OSM overlay underneath."""
    bki_path = Path(args.bki)
    if not bki_path.exists():
        raise SystemExit(f"File not found: {args.bki}")
    if not Path(args.config).exists():
        raise SystemExit(f"Config not found: {args.config}")

    viz_cfg = load_visualize_config(args.config)
    colors_from_config = viz_cfg['colors_by_label'] if viz_cfg['colors_by_label'] else None
    osm_path = None if args.no_osm else (args.osm or viz_cfg['osm_path'])

    print(f"Loading {bki_path}...")
    points_xyz, labels_raw = read_bki(str(bki_path), args.config, min_alpha_sum=args.min_alpha, min_max_prob=args.min_max_prob)
    n = len(points_xyz)
    if n == 0:
        raise SystemExit("No voxels to show (try lowering --min-alpha or --min-max-prob).")

    colors = get_label_colors(labels_raw, args.format, colors_from_config=colors_from_config)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"BKI Map - {bki_path.name}", width=1280, height=720)

    # Add OSM underneath first (if configured and file exists; uses C++ loader)
    osm_shown = False
    if osm_path and Path(osm_path).exists():
        try:
            loader = OSMLoader(osm_path, args.config)
            osm_geoms = loader.get_geometries(z_offset=args.osm_z_offset, use_thick=args.osm_thick, thickness=args.osm_thickness)
            for g in osm_geoms:
                vis.add_geometry(g)
            osm_shown = True
        except ImportError as e:
            print(f"Warning: {e}; skipping OSM overlay.")
        except Exception as e:
            print(f"Warning: OSM load failed: {e}; skipping OSM overlay.")

    vis.add_geometry(pcd)
    vis.get_render_option().point_size = args.point_size
    vis.get_render_option().background_color = np.array([0.05, 0.05, 0.05])

    print(f"Showing {n} voxels" + (" + OSM" if osm_shown else "") + ". Close the window to exit.")
    vis.run()
    vis.destroy_window()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="BKI map tools: convert and visualize .bki files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bki_tools convert map.bki --config mcd_config.yaml
  bki_tools visualize map.bki --config mcd_config.yaml --min-alpha 0.01 --min-max-prob 0.5
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand")

    # convert
    p_convert = subparsers.add_parser("convert", help="Export .bki to .bin and .label files")
    p_convert.add_argument("bki", help="Path to .bki map file")
    p_convert.add_argument("--config", required=True, help="Path to YAML config (same as used when building the map)")
    p_convert.add_argument("--output-bin", default=None, help="Output .bin path (default: <bki_stem>_map.bin)")
    p_convert.add_argument("--output-label", default=None, help="Output .label path (default: <bki_stem>_map.label)")
    p_convert.add_argument("--min-alpha", type=float, default=1e-6, help="Min alpha sum per voxel (default: 1e-6)")
    p_convert.add_argument("--min-max-prob", type=float, default=0.0, help="Min predicted class probability max(alpha)/sum(alpha) (default: 0.0)")
    p_convert.set_defaults(func=cmd_convert)

    # visualize
    p_vis = subparsers.add_parser("visualize", help="Display .bki voxels in Open3D viewer")
    p_vis.add_argument("bki", help="Path to .bki map file")
    p_vis.add_argument("--config", required=True, help="Path to YAML config used when building the map")
    p_vis.add_argument("--min-alpha", type=float, default=1e-6, help="Min alpha sum per voxel (default: 1e-6)")
    p_vis.add_argument("--min-max-prob", type=float, default=0.0, help="Min predicted class probability max(alpha)/sum(alpha) (default: 0.0)")
    p_vis.add_argument("--format", choices=["auto", "mcd", "kitti"], default="auto", help="Label format for colors (ignored when config has colors)")
    p_vis.add_argument("--point-size", type=float, default=2.0, help="Point size (default: 2.0)")
    p_vis.add_argument("--osm-z-offset", type=float, default=0.05, help="Z height for OSM lines (default: 0.05)")
    p_vis.add_argument("--osm-thick", action="store_true", help="Render OSM as thick cylinder meshes")
    p_vis.add_argument("--osm-thickness", type=float, default=10.0, help="Cylinder radius when --osm-thick (default: 10.0)")
    p_vis.add_argument("--osm", type=str, default=None, help="Override OSM file path (from config by default)")
    p_vis.add_argument("--no-osm", action="store_true", help="Disable OSM overlay even if config has osm_file")
    p_vis.set_defaults(func=cmd_visualize)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
