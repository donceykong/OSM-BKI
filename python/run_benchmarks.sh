#!/usr/bin/env bash
# Run all OSM-BKI benchmarks. Works from any directory: ./python/run_benchmarks.sh or cd python && ./run_benchmarks.sh
# Requires: conda activate composite-bki (or equivalent env with composite_bki_cpp, scipy, pandas)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Paths relative to benchmarks/ (each script resolves from its own directory)
data_dir="../../example_data/kth_day_06"
scan_dir="$data_dir/lidar_bin/data"
pred_labels="$data_dir/labels_predicted/labels_predicted"
gt_labels="$data_dir/gt_labels"
pose_path="$data_dir/pose_inW.csv"
osm_path="../../example_data/kth.osm"
config_path="../../configs/mcd_config.yaml"

run() {
  echo "=== $1 ==="
  python "benchmarks/$1" "${@:2}"
}

run throughput_benchmark.py \
  --scan-dir "$scan_dir" \
  --label-dir "$pred_labels" \
  --osm "$osm_path" --config "$config_path"

run calibration_benchmark.py \
  --scan-dir "$scan_dir" \
  --label-dir "$pred_labels" \
  --gt-dir "$gt_labels" \
  --osm "$osm_path" --pose "$pose_path" --config "$config_path"

run forgetting_benchmark.py \
  --scan-dir "$scan_dir" \
  --label-dir "$pred_labels" \
  --gt-dir "$gt_labels" \
  --osm "$osm_path" --pose "$pose_path" --config "$config_path"

run multi_scan_benchmark.py \
  --scan-dir "$scan_dir" \
  --label-dir "$pred_labels" \
  --gt-dir "$gt_labels" \
  --osm "$osm_path" --pose "$pose_path" --config "$config_path"

run osm_modes_benchmark.py \
  --scan-dir "$scan_dir" \
  --label-dir "$pred_labels" \
  --gt-dir "$gt_labels" \
  --osm "$osm_path" --pose "$pose_path" --config "$config_path"

run parameter_sensitivity_benchmark.py \
  --scan-dir "$scan_dir" \
  --label-dir "$pred_labels" \
  --gt-dir "$gt_labels" \
  --osm "$osm_path" --pose "$pose_path" --config "$config_path"

run pred_vs_gt_benchmark.py \
  --scan-dir "$scan_dir" \
  --pred-dir "$pred_labels" \
  --gt-dir "$gt_labels" \
  --osm "$osm_path" --pose "$pose_path" --config "$config_path"

run temporal_consistency_benchmark.py \
  --scan-dir "$scan_dir" \
  --label-dir "$pred_labels" \
  --osm "$osm_path" --pose "$pose_path" --config "$config_path"

run per_class_iou_benchmark.py \
  --scan-dir "$scan_dir" \
  --label-dir "$pred_labels" \
  --gt-dir "$gt_labels" \
  --osm "$osm_path" --pose "$pose_path" --config "$config_path"

run kernel_ablation_benchmark.py \
  --scan-dir "$scan_dir" \
  --gt-dir "$gt_labels" \
  --osm "$osm_path" --config "$config_path"

run noise_benchmark.py \
  --scan-dir "$scan_dir" \
  --gt-dir "$gt_labels" \
  --osm "$osm_path" --config "$config_path"

run osm_randomization_benchmark.py \
  --lidar "$scan_dir/0000000011.bin" \
  --labels "$pred_labels/0000000011.bin" \
  --gt-labels "$gt_labels/0000000011.bin" \
  --osm "$osm_path" \
  --config "$config_path"

echo "Done."
