#!/usr/bin/env python3
"""
Python port of visualize_map_osm.cpp.

Accumulates a labelled LiDAR map from all frames in a MCD sequence and
overlays OSM polylines, using the same transform chain as the C++ tool:

    lidar_to_map = body_to_world_shifted * inv(body_to_lidar)

where  body_to_world_shifted = poseToMatrix(pose) – init_rel_pos.

Usage (config-file driven, mirrors C++ positional args):
    python visualize_osm_xml.py --config configs/mcd_config.yaml

Usage (explicit paths, no config required):
    python visualize_osm_xml.py \\
        --osm       kth.osm \\
        --scan_dir  kth_day_06/lidar_bin/data \\
        --label_dir kth_day_06/gt_labels \\
        --pose      kth_day_06/pose_inW.csv \\
        --calib     hhs_calib.yaml \\
        --osm_origin_lat 59.348268650 \\
        --osm_origin_lon 18.073204280 \\
        --init_rel_pos 64.393 66.483 38.514 \\
        --skip_frames 10 \\
        --max_scans 100
"""

import argparse
from typing import Optional
import os
import sys
import yaml
from pathlib import Path

import numpy as np
import open3d as o3d
import composite_bki_cpp

from osm_loader import OSMLoader, create_thick_lines

# Semantic label colours (from mcd_config.yaml `colors:` block, normalised to [0,1])
MCD_LABEL_COLORS = {
    0:  [128/255, 128/255, 128/255],  # barrier
    1:  [119/255,  11/255,  32/255],  # bike
    2:  [  0/255, 100/255,   0/255],  # building
    3:  [139/255,  69/255,  19/255],  # chair
    4:  [101/255,  67/255,  33/255],  # cliff
    5:  [160/255, 160/255, 160/255],  # container
    6:  [244/255,  35/255, 232/255],  # curb
    7:  [190/255, 153/255, 153/255],  # fence
    8:  [255/255, 165/255,   0/255],  # hydrant
    9:  [255/255, 255/255,   0/255],  # infosign
    10: [170/255, 255/255, 150/255],  # lanemarking
    11: [  0/255,   0/255,   0/255],  # noise
    12: [255/255, 255/255,  50/255],  # other
    13: [250/255, 170/255, 160/255],  # parkinglot
    14: [220/255,  20/255,  60/255],  # pedestrian
    15: [153/255, 153/255, 153/255],  # pole
    16: [128/255,  64/255, 128/255],  # road
    17: [  0/255, 100/255,   0/255],  # shelter
    18: [244/255,  35/255, 232/255],  # sidewalk
    19: [128/255,   0/255, 128/255],  # stairs
    20: [  0/255, 150/255, 255/255],  # structure-other
    21: [255/255,  69/255,   0/255],  # traffic-cone
    22: [  0/255,   0/255, 255/255],  # traffic-sign
    23: [139/255,   0/255, 139/255],  # trashbin
    24: [  0/255,  60/255, 135/255],  # treetrunk
    25: [107/255, 142/255,  35/255],  # vegetation
    26: [245/255, 150/255, 100/255],  # vehicle-dynamic
    27: [ 51/255,   0/255,  51/255],  # vehicle-other
    28: [  0/255,   0/255, 142/255],  # vehicle-static
}

# Vectorised look-up table: index = label id, value = [r, g, b] float32
_MAX_LABEL = max(MCD_LABEL_COLORS) + 1
_COLOR_LUT  = np.zeros((_MAX_LABEL, 3), dtype=np.float32)
for _lbl, _rgb in MCD_LABEL_COLORS.items():
    _COLOR_LUT[_lbl] = _rgb


def _color_from_label(label: int) -> list:
    """Return [r, g, b] in [0, 1]; MCD lookup first, then hash fallback
    matching colorFromLabel() in visualize_map_osm.cpp."""
    if 0 <= label < _MAX_LABEL:
        return _COLOR_LUT[label].tolist()
    h = (label * 2654435761) & 0xFFFFFFFF
    return [((h >> 16) & 0xFF) / 255.0,
            ((h >>  8) & 0xFF) / 255.0,
            ( h        & 0xFF) / 255.0]


# ─── Dataset / calibration loading ───────────────────────────────────────────

def load_dataset_config(config_path: str) -> dict:
    """
    Read mcd_config.yaml and return a dict mirroring DatasetConfig in
    dataset_utils.hpp, with all paths fully resolved.

    Mirrors loadDatasetConfig() + path construction in dataset_utils.cpp.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    root = raw['dataset_root_path']
    seq  = raw['sequence']
    base = os.path.join(root, seq)

    cfg: dict = {
        'skip_frames':       int(raw.get('skip_frames', 0)),
        'lidar_dir':         os.path.join(base, 'lidar_bin', 'data'),
        'label_dir':         os.path.join(base, 'gt_labels'),
        'pose_path':         os.path.join(base, 'pose_inW.csv'),
        'calibration_path':  os.path.join(root, 'hhs_calib.yaml'),
        'osm_file':          '',
        'use_osm_origin':    False,
        'osm_origin_lat':    0.0,
        'osm_origin_lon':    0.0,
        'use_init_rel_pos':  False,
        'init_rel_pos':      np.zeros(3),
        'colors_by_label':   {},
    }

    if 'osm_file' in raw:
        cfg['osm_file'] = os.path.join(root, raw['osm_file'])

    # init_latlon_day_06 → OSM projection origin (matches use_osm_origin_from_mcd)
    if 'init_latlon_day_06' in raw:
        ll = raw['init_latlon_day_06']
        cfg['osm_origin_lat'] = float(ll[0])
        cfg['osm_origin_lon'] = float(ll[1])
        cfg['use_osm_origin'] = True

    # init_rel_pos_day_06 → pose re-centring (matches use_init_rel_pos)
    if 'init_rel_pos_day_06' in raw:
        rp = raw['init_rel_pos_day_06']
        cfg['init_rel_pos']     = np.array([float(rp[0]), float(rp[1]), float(rp[2])])
        cfg['use_init_rel_pos'] = True

    # Semantic label colours from config `colors:` block
    if 'colors' in raw:
        for lbl_str, rgb in raw['colors'].items():
            cfg['colors_by_label'][int(lbl_str)] = [c / 255.0 for c in rgb]

    return cfg


def load_body_to_lidar(calib_yaml_path: str) -> np.ndarray:
    """
    Read body/os_sensor/T from hhs_calib.yaml → 4×4 float64 matrix.
    Mirrors readBodyToLidarCalibration() in file_io.cpp.
    """
    with open(calib_yaml_path) as f:
        calib = yaml.safe_load(f)
    rows = calib['body']['os_sensor']['T']
    mat  = np.array(rows, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(
            f"Expected 4×4 matrix in {calib_yaml_path}, got {mat.shape}")
    print(f"Loaded body→LiDAR calibration from {calib_yaml_path}")
    return mat


def load_poses_csv(pose_csv_path: str) -> dict:
    """
    Read pose CSV (num, t, x, y, z, qx, qy, qz[, qw]) and return
    {scan_id: np.array([x, y, z, qx, qy, qz, qw])}.

    Mirrors readPosesCSV() / parsePoseLine() in file_io.cpp:
      col 0 → scan_id   col 2-4 → x, y, z
      col 5-7 → qx, qy, qz   col 8 → qw (default 1.0 if absent)
    Non-numeric tokens (CSV header) are silently skipped.
    """
    poses: dict = {}
    with open(pose_csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            sep  = ',' if ',' in line else None
            vals = []
            for tok in (line.split(sep) if sep else line.split()):
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
            if len(vals) < 8:
                continue
            scan_id       = int(vals[0])
            qw            = vals[8] if len(vals) > 8 else 1.0
            poses[scan_id] = np.array(
                [vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], qw],
                dtype=np.float64)
    print(f"Loaded {len(poses)} poses from {pose_csv_path}")
    return poses


def collect_scan_label_pairs(lidar_dir: str, label_dir: str) -> list:
    """
    Return a sorted list of (scan_id, scan_path, label_path) for all .bin
    files in lidar_dir that have a matching .bin in label_dir.
    Mirrors collectScanLabelPairs() in dataset_utils.cpp.
    """
    pairs = []
    for lf in sorted(Path(lidar_dir).glob('*.bin')):
        lp = Path(label_dir) / (lf.stem + '.bin')
        if lp.exists():
            try:
                sid = int(lf.stem)
            except ValueError:
                sid = -1
            pairs.append((sid, str(lf), str(lp)))
    print(f"Found {len(pairs)} scan/label pairs in {lidar_dir}")
    return pairs


def transform_scan_to_world(points: np.ndarray,
                            pose: np.ndarray,
                            body_to_lidar: np.ndarray,
                            init_rel_pos: np.ndarray) -> np.ndarray:
    """Delegate to C++ pose_utils::transformScanToWorld (shared with visualize_map_osm.cpp)."""
    irp = np.asarray(init_rel_pos, dtype=np.float64) if np.any(init_rel_pos) else None
    return composite_bki_cpp.transform_scan_to_world(
        np.ascontiguousarray(points, dtype=np.float32),
        np.asarray(pose, dtype=np.float64),
        np.ascontiguousarray(body_to_lidar, dtype=np.float64),
        irp,
    )


def build_map_cloud(pairs: list,
                    poses_by_id: dict,
                    body_to_lidar: np.ndarray,
                    init_rel_pos: np.ndarray,
                    colors_by_label: dict,
                    step: int = 1,
                    max_scans: Optional[int] = None) -> 'o3d.geometry.PointCloud | None':
    """
    Accumulate scans (stride=step) into a single Open3D PointCloud coloured
    by semantic label.  Mirrors the scan loop in visualize_map_osm.cpp.

    Binary format: lidar .bin = float32[N*4] (x,y,z,intensity),
                   label .bin = uint32[N].
    """
    all_pts    = []
    all_colors = []
    loaded = skipped = 0

    for i in range(0, len(pairs), step):
        if max_scans is not None and loaded >= max_scans:
            break
        scan_id, scan_path, label_path = pairs[i]
        if scan_id not in poses_by_id:
            skipped += 1
            continue

        try:
            raw = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
            pts = raw[:, :3]
        except Exception as e:
            print(f"  Warning: could not read {scan_path}: {e}")
            skipped += 1
            continue

        try:
            labels = np.fromfile(label_path, dtype=np.uint32)
        except Exception as e:
            print(f"  Warning: could not read {label_path}: {e}")
            skipped += 1
            continue

        if len(labels) != len(pts):
            skipped += 1
            continue

        world = transform_scan_to_world(
            pts, poses_by_id[scan_id], body_to_lidar, init_rel_pos)

        # Colour by label (vectorised per unique label)
        colours = np.zeros((len(labels), 3), dtype=np.float32)
        for lbl in np.unique(labels):
            mask = labels == lbl
            if int(lbl) in colors_by_label:
                colours[mask] = colors_by_label[int(lbl)]
            else:
                colours[mask] = _color_from_label(int(lbl))

        all_pts.append(world)
        all_colors.append(colours)
        loaded += 1

    print(f"Scans loaded: {loaded}, skipped: {skipped}")
    if not all_pts:
        return None

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    cloud.colors = o3d.utility.Vector3dVector(
        np.vstack(all_colors).astype(np.float64))
    return cloud


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Python port of visualize_map_osm.cpp – accumulates a "
                    "labelled LiDAR map and overlays OSM polylines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    # ── Config file (mirrors C++ first two positional args) ────────────────
    parser.add_argument("--config", type=str, default=None,
                        help="Path to mcd_config.yaml (resolves all paths "
                             "automatically, matching C++ loadDatasetConfig).")
    parser.add_argument("--skip_frames", type=int, default=None,
                        help="Override skip_frames from config "
                             "(C++ third positional arg; default: use config or 0).")
    parser.add_argument("--max_scans", type=int, default=None,
                        help="Maximum number of scans to add to the map (default: no limit).")

    # ── Explicit path overrides ────────────────────────────────────────────
    parser.add_argument("--osm", type=str, default=None,
                        help="Path to .osm file.")
    parser.add_argument("--scan_dir", type=str, default=None,
                        help="Directory containing lidar .bin files "
                             "(dataset_root/sequence/lidar_bin/data).")
    parser.add_argument("--label_dir", type=str, default=None,
                        help="Directory containing label .bin files "
                             "(dataset_root/sequence/gt_labels).")
    parser.add_argument("--pose", type=str, default=None,
                        help="Path to pose_inW.csv.")
    parser.add_argument("--calib", type=str, default=None,
                        help="Path to hhs_calib.yaml (body→LiDAR calibration).")
    parser.add_argument("--init_rel_pos", type=float, nargs=3,
                        metavar=('X', 'Y', 'Z'), default=None,
                        help="World-frame initial position to subtract from "
                             "poses (init_rel_pos_day_06 in mcd_config.yaml).")
    parser.add_argument("--osm_origin_lat", type=float, default=None,
                        help="Override OSM projection origin latitude "
                             "(init_latlon_day_06[0]; KTH: 59.348268650).")
    parser.add_argument("--osm_origin_lon", type=float, default=None,
                        help="Override OSM projection origin longitude "
                             "(init_latlon_day_06[1]; KTH: 18.073204280).")
    parser.add_argument("--osm_world_offset_x", type=float, default=0.0,
                        help="World-frame X offset for the OSM origin (default 0.0).")
    parser.add_argument("--osm_world_offset_y", type=float, default=0.0,
                        help="World-frame Y offset for the OSM origin (default 0.0).")

    # ── Visualisation tweaks ───────────────────────────────────────────────
    parser.add_argument("--z_offset", type=float, default=0.05,
                        help="Z height for OSM lines (default 0.05, matching C++).")
    parser.add_argument("--thick", action="store_true",
                        help="Render OSM as thick cylinder meshes (slow on large maps).")
    parser.add_argument("--thickness", type=float, default=20.0,
                        help="Cylinder radius in metres when --thick is active.")
    args = parser.parse_args()

    # ── Resolve configuration ──────────────────────────────────────────────
    cfg: dict = {
        'skip_frames':      0,
        'lidar_dir':        None,
        'label_dir':        None,
        'pose_path':        None,
        'calibration_path': None,
        'osm_file':         '',
        'use_osm_origin':   False,
        'osm_origin_lat':   0.0,
        'osm_origin_lon':   0.0,
        'use_init_rel_pos': False,
        'init_rel_pos':     np.zeros(3),
        'colors_by_label':  {},
    }

    if args.config:
        cfg.update(load_dataset_config(args.config))
        print(f"Loaded config from {args.config}")

    # CLI args always override config file values
    if args.skip_frames  is not None:  cfg['skip_frames']      = args.skip_frames
    if args.scan_dir:                  cfg['lidar_dir']         = args.scan_dir
    if args.label_dir:                 cfg['label_dir']         = args.label_dir
    if args.pose:                      cfg['pose_path']         = args.pose
    if args.calib:                     cfg['calibration_path']  = args.calib
    if args.osm:                       cfg['osm_file']          = args.osm
    if args.init_rel_pos is not None:
        cfg['init_rel_pos']     = np.array(args.init_rel_pos)
        cfg['use_init_rel_pos'] = True
    if args.osm_origin_lat is not None and args.osm_origin_lon is not None:
        cfg['osm_origin_lat'] = args.osm_origin_lat
        cfg['osm_origin_lon'] = args.osm_origin_lon
        cfg['use_osm_origin'] = True

    step = cfg['skip_frames'] + 1

    # ── Validate required inputs ───────────────────────────────────────────
    missing = [k for k in ('lidar_dir', 'label_dir', 'pose_path',
                           'calibration_path', 'osm_file')
               if not cfg.get(k)]
    if missing:
        parser.error(
            f"Missing required inputs: {missing}. "
            "Provide --config or the individual path flags.")

    # ── Load poses ─────────────────────────────────────────────────────────
    poses_by_id = load_poses_csv(cfg['pose_path'])
    if not poses_by_id:
        sys.exit("No poses loaded.")

    # ── Load calibration ───────────────────────────────────────────────────
    body_to_lidar = load_body_to_lidar(cfg['calibration_path'])
    init_rel_pos  = cfg['init_rel_pos'] if cfg['use_init_rel_pos'] else np.zeros(3)

    print(f"init_rel_pos = [{init_rel_pos[0]:.4f}, "
          f"{init_rel_pos[1]:.4f}, {init_rel_pos[2]:.4f}]")

    # ── Collect scan/label pairs ───────────────────────────────────────────
    pairs = collect_scan_label_pairs(cfg['lidar_dir'], cfg['label_dir'])
    if not pairs:
        sys.exit("No scan/label pairs found.")

    # ── Accumulate map cloud (C++ transform_scan_to_world per scan) ───────
    print(f"Accumulating map (step={step}, pairs={len(pairs)}"
          + (f", max_scans={args.max_scans})" if args.max_scans else ")") + "...")
    map_cloud = build_map_cloud(
        pairs, poses_by_id, body_to_lidar, init_rel_pos,
        cfg['colors_by_label'],
        step=step, max_scans=args.max_scans)
    if map_cloud is None or len(map_cloud.points) == 0:
        sys.exit("No map points accumulated.")
    print(f"Map cloud: {len(map_cloud.points):,} points")

    # ── Load OSM (uses C++ loader; requires --config) ──────────────────────
    if args.config:
        loader = OSMLoader(cfg['osm_file'], args.config)
        osm_geoms = loader.get_geometries(
            z_offset=args.z_offset, use_thick=args.thick, thickness=args.thickness)
    else:
        print("Warning: OSM overlay requires --config; skipping OSM.")
        osm_geoms = []

    # ── Summary (mirrors C++ stdout) ───────────────────────────────────────
    n_osm = sum(len(g.lines) for g in osm_geoms if hasattr(g, 'lines'))
    print(f"\nMap points={len(map_cloud.points):,}, "
          f"skip_frames={cfg['skip_frames']}, "
          f"initial_position_xyz=[{init_rel_pos[0]:.3f}, "
          f"{init_rel_pos[1]:.3f}, {init_rel_pos[2]:.3f}], "
          f"OSM segments={n_osm}")

    # ── Visualise (matches PCLVisualizer setup in visualize_map_osm.cpp) ──
    print("\nLaunching visualisation…")
    print("  Mouse: rotate / zoom / pan   R: reset view   Q / Esc: quit")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Map + OSM Viewer", width=1280, height=720)

    vis.add_geometry(map_cloud)
    for g in osm_geoms:
        vis.add_geometry(g)

    # Coordinate axes (matches viewer->addCoordinateSystem(1.0) in C++)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=5.0, origin=[0, 0, 0])
    vis.add_geometry(axes)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])  # matches C++ 0.05,0.05,0.05
    opt.point_size = 2.0                                  # matches PCL_VISUALIZER_POINT_SIZE 2

    vis.poll_events()
    vis.update_renderer()
    vis.reset_view_point(True)

    ctr = vis.get_view_control()
    ctr.set_constant_z_far(1_000_000)
    ctr.set_constant_z_near(0.1)
    ctr.set_front([0, 0, -1])   # top-down view
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.3)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
