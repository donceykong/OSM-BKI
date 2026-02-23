# osm_bki (C++)

C++ library and tools for OSM-Enhanced Semantic Bayesian Kernel Inference (OSM-SBKI). Here you will find the static library `osm_bki_core` and an optional PCL-based visualization example.

## Requirements

- **CMake** ≥ 3.16  
- **C++17** compiler (e.g. GCC 8+, Clang 6+)  
- **Eigen3** (for core library)  
- **OpenMP** (optional; used when available)

To build the example program `visualize_map_osm` you also need:

- **MPI** (C component)

- **PCL** (Point Cloud Library) with components: `common`, `io`, `visualization`

- **libosmium** (header-only; for reading OSM files) and its runtime deps (expat, bz2, zlib for OSM input)

### Installing dependencies (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install -y build-essential cmake libeigen3-dev libpcl-dev libosmium-dev libopenmpi-dev libexpat1-dev libbz2-dev zlib1g-dev
```

If libosmium is not in your package manager, set `OSMIUM_INCLUDE_DIR` when configuring (see Build below).

### Installing dependencies (MacOS)
```bash
brew install cmake eigen pcl libosmium open-mpi libomp expat bzip2
```

- **Eigen** and **expat** are optional if you only build the core library; **libomp** provides OpenMP on Apple Clang.
- For the `visualize_map_osm` example you need **PCL**, **libosmium**, **open-mpi**, and the rest as above.

If libosmium is not found by CMake, set `OSMIUM_INCLUDE_DIR` when configuring (e.g. `$(brew --prefix libosmium)/include`). See Build below.

## Project structure

```
osm_bki/
├── CMakeLists.txt
├── include/           # Public headers
│   ├── continuous_bki.hpp
│   ├── dataset_utils.hpp
│   ├── file_io.hpp
│   ├── osm_parser.hpp
│   ├── osm_xml_parser.hpp
│   └── yaml_parser.hpp
├── src/               # Core library and OSM parser
│   ├── continuous_bki.cpp
│   ├── dataset_utils.cpp
│   ├── file_io.cpp
│   └── osm_parser.cpp
├── examples/
│   └── visualize_map_osm.cpp   # PCL viewer: map + OSM polylines
└── configs/
    ├── mcd_config.yaml        # MCD dataset (paths, OSM file, origins)
    ├── osm_config.yaml       # OSM category filters
    ├── kitti360_config.yaml
    └── kitti_config.yaml
```

## Build

From the `osm_bki` directory (where this README and `CMakeLists.txt` live):

**Library only (default):**

```bash
cmake -S . -B build
cmake --build build
```

**Library + example (`visualize_map_osm`):**

```bash
cmake -S . -B build -Dosm_bki_BUILD_PCL_EXAMPLES=ON
cmake --build build
```

If libosmium is installed in a non-standard location:

```bash
cmake -S . -B build -Dosm_bki_BUILD_PCL_EXAMPLES=ON -DOSMIUM_INCLUDE_DIR=/path/to/osmium/include
cmake --build build
```

The example executable is produced at `build/visualize_map_osm`.

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `osm_bki_BUILD_PCL_EXAMPLES` | `OFF` | Build the `visualize_map_osm` example (requires PCL and libosmium). |
| `OSMIUM_INCLUDE_DIR` | (empty) | Path to libosmium include directory if not found automatically. |
| `osm_bki_ENABLE_WARNINGS` | `ON` | Enable compiler warnings. |

## Running the example (visualize_map_osm)

`visualize_map_osm` loads an MCD-style dataset (lidar scans, labels, poses), builds a colored point cloud in world frame, parses an OSM file into 2D polylines, and shows both in a PCL viewer.

**Usage:**

```bash
./build/visualize_map_osm [mcd_config.yaml] [osm_config.yaml] [skip_frames]
```

- **mcd_config.yaml** – Dataset and OSM settings (default: `configs/mcd_config.yaml`).
- **osm_config.yaml** – OSM category toggles (default: `configs/osm_config.yaml`).
- **skip_frames** – Optional; overrides `skip_frames` in the MCD config.

**Example:**

```bash
./build/visualize_map_osm configs/mcd_config.yaml configs/osm_config.yaml
```

### MCD config (mcd_config.yaml)

Key fields used by the example:

**Path resolution (choose one):**
- **Direct paths** (relative to config file): `lidar_dir`, `label_dir`, `pose_path`, `calibration_path`, `osm_file`.
å
**Other fields:**
- **init_latlon_day_06** – `[lat, lon]` used as OSM local coordinate origin.
- **init_rel_pos_day_06** – `[x, y, z]` world origin for aligning poses (subtracted from pose positions).
- **skip_frames** – Number of frames to skip between scans used for the map.

### OSM config (osm_config.yaml)

Boolean flags to include or exclude OSM categories: buildings, roads, sidewalks, parking, fences, stairs, grasslands, trees. Used only when building the example; the core library does not depend on this file.

## License

See the top-level OSM-S-BKI repository license.
