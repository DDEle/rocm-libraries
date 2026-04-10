# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Composable Kernel (CK)** is an AMD ROCm library for high-performance GPU kernels targeting ML workloads. It provides a tile-based programming model in C++17/20 with HIP for operations like GEMM, convolution, Flash Attention, normalization, and more. Supported GPU architectures: gfx908, gfx90a, gfx942, gfx950, gfx1030, gfx1100–1103, gfx1200/1201.

## Build Commands

```bash
# Dev build — always use this form (do NOT use cmake-ck-dev.sh script)
mkdir build-<arch> && cd build-<arch>
cmake .. \
  -GNinja \
  -B build-<arch> \
  --preset dev \
  -D GPU_TARGETS="<arch>"   # replace with target arch, e.g. gfx950
ninja -j$(nproc) <target>

# Shorthand: single-GPU presets (dev-gfx908, dev-gfx90a, dev-gfx942, dev-gfx950)
cmake .. -GNinja --preset dev-<arch>
ninja -j$(nproc) <target>

# Standard (non-dev) build — uses hipcc, no -Werror
mkdir build-<arch> && cd build-<arch>
cmake \
  -GNinja \
  -B build-<arch> \
  -D CMAKE_PREFIX_PATH=/opt/rocm \
  -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -D CMAKE_BUILD_TYPE=Release \
  -D GPU_TARGETS="<arch>" \
  ..
ninja -j$(nproc)

# Install
ninja -j$(nproc) install
```

**Important**: Always pass `-GNinja` explicitly on the cmake command line even when using `--preset`, as CMakePresets.json does not set a generator. The `dev` preset uses `/opt/rocm/llvm/bin/clang++` (not `hipcc`) and enables `BUILD_DEV=ON` (`-Werror -Weverything`).

Key CMake options:
- `GPU_TARGETS` — semicolon-separated GPU targets (e.g. `"gfx942;gfx90a"`)
- `DTYPES` — subset of `fp64;fp32;tf32;fp16;fp8;bf16;int8`
- `BUILD_DEV` — enable `-Werror -Weverything` (default ON in dev preset)
- `CK_TIME_KERNEL` — enable kernel timing (default ON)
- `DISABLE_DL_KERNELS` / `DISABLE_DPP_KERNELS` — skip specific kernel families
- `BUILD_MHA_LIB` — build static Flash Attention library
- `CK_CXX_STANDARD` — C++ standard, 17 or 20 (default 20)
- `CK_USE_FP8_ON_UNSUPPORTED_ARCH` — enable fp8 on older architectures

## Testing

```bash
# Build and run all tests
make -j check

# Build all tests without running
make -j tests

# Run smoke tests only (< 30s each)
make -j smoke

# Run regression tests (>= 30s each)
make -j regression

# Run a single test by name pattern (from build directory)
ctest -R test_gemm_fp16

# Run test binary directly
./bin/test_gemm_fp16
```

Tests use Google Test (fetched via CMake `FetchContent`). Tests under `test/` are labeled `SMOKE_TEST` or `REGRESSION_TEST` via a hardcoded list in `test/CMakeLists.txt`. The `add_test_executable()` and `add_gtest_executable()` CMake functions define tests. Some tests are automatically excluded based on GPU architecture (e.g., XDL tests excluded for non-gfx9/gfx11/gfx12 targets, WMMA tests excluded for non-gfx11/gfx12).

## Linting and Formatting

```bash
# Run all pre-commit checks
pre-commit run --all-files

# Install pre-commit hooks (first time)
sudo script/install_precommit.sh

# Format C++ files in-place with clang-format
script/clang-format-overwrite.sh
```

Pre-commit hooks enforce: clang-format v18 (C++/.inc files), ruff (Python), copyright headers (C++, Python, shell, CMake), and executable bit removal. Config: `.clang-format` (100-col limit, 4-space indent, Allman-style braces), `.clang-tidy` (static analysis enabled in CMake via `ENABLE_CLANG_CPP_CHECKS`).

## Architecture

CK has two parallel programming models:

### Classic CK (`include/ck/`)
Four-layer abstraction from low to high:
1. **Tile Operators** (`include/ck/tensor_operation/gpu/`) — thread/warp/block/grid-level primitives
2. **Device Kernels** (`include/ck/tensor_operation/gpu/device/`) — abstract device operation interfaces and template implementations
3. **Kernel Instances** (`library/src/tensor_operation_instance/gpu/`) — pre-instantiated templates per dtype/layout/GPU
4. **Client API** (`library/include/ck/library/`) — instance factory for selecting and running kernels

### CK Tile (`include/ck_tile/`)
The newer, independent tile-based model:
- **core/** — containers (array, tuple, sequence), numeric types (fp16, bf16, fp8, int8), tensor descriptors, coordinate transforms, distributed tensors, tile-level APIs (`load_tile`, `store_tile`, `shuffle_tile`)
- **ops/** — operator implementations: gemm, fmha (Flash Attention), convolution, layernorm2d, rmsnorm2d, softmax, reduce, pooling, permute, elementwise, topk, fused_moe, sparse_attn, smoothquant, batched_contraction
- **host/** — host-side utilities for kernel launching and device buffers
- **ref/** — CPU/GPU reference implementations for testing

### Other Components
- **python/ck4inductor/** — PyTorch Inductor integration (installed as `rocm-composable-kernel` Python package via `pyproject.toml`); includes `universal_gemm`, `batched_universal_gemm`, `ck_tile_universal_gemm`, `grouped_conv_fwd`
- **experimental/builder/** — experimental builder system

### CK Tile Examples (`example/ck_tile/`)
Numbered example directories (non-contiguous — gaps indicate removed/reserved slots):
`01_fmha`, `02_layernorm2d`, `03_gemm`, `04_img2col`, `05_reduce`, `06_permute`, `09_topk_softmax`, `10_rmsnorm2d`, `11_add_rmsnorm2d_rdquant`, `12_smoothquant`, `13_moe_sorting`, `14_moe_smoothquant`, `15_fused_moe`, `16_batched_gemm`, `17_grouped_gemm`, `18_flatmm`, `19_gemm_multi_d`, `20_grouped_convolution`, `21_elementwise`, `22_gemm_multi_abd`, `35_batched_transpose`, `36_pooling`, `37_transpose`, `38_block_scale_gemm`, `40_streamk_gemm`, `41_batched_contraction`.

Each example directory typically contains: a `generate.py` + `codegen/ops/<op>.py` for instance generation, a `<op>.hpp` kernel implementation, an `example_<op>.cpp` host driver, and a `CMakeLists.txt`. Generated instance files land in `build/example/ck_tile/<dir>/` (not in the source tree).

### Key Terminology
See `TERMINOLOGY.md` for definitions of hardware terms (wavefront, LDS, MFMA), programming model terms (tile, descriptor, transform), and operation abbreviations. See `ACRONYMS.md` for acronym definitions.

## CI/CD
Jenkins is the primary CI system (`Jenkinsfile`). There are no GitHub Actions workflows. Docker images are provided (see `Dockerfile*`) for reproducible dev environments including ROCm and sccache support.

## Docker Workflow for Build and Run

**每次编译或运行项目时，必须遵循以下流程：**

1. **创建新容器**，使用以下参数：
   ```bash
   docker run --rm --privileged --group-add sudo --net host \
     --device /dev/kfd --device /dev/dri \
     -v /home/yiding12/workspace:/home/yiding12/workspace \
     -w /home/yiding12/workspace/rocm-libraries/projects/composablekernel \
     --name yiding12-ck-<unique-suffix> \
     <IMAGE> \
     bash -c "<命令>" \
     > /home/yiding12/workspace/rocm-libraries/projects/composablekernel/compile.log 2>&1
   ```
   **注意**：
   - `--device /dev/kfd --device /dev/dri` 是运行 GPU kernel 必须的；纯编译可省略。
   - **编译 log 输出到项目目录**（宿主机路径），不要输出到 `/tmp`。
   - **docker 命令本身不截断输出**（不在 `bash -c "..."` 内部加 `| tail`）；读 log 文件时再用 `tail`。
   - **启动前先检查同名容器**：若容器名已存在（上次被中断未自动清理），先 `docker rm -f <name>` 再启动。
   - **编译并行数限制为总核心数的 80%**。

2. **默认镜像**：
   ```
   registry-sc-harbor.amd.com/framework/therock-main:1158_gfx950_7.13.0a20260331_ubuntu24.04_py3.12_pytorch_release-2.10_1f8cea47f2
   ```
   除非另有指定，否则始终使用此镜像。

   容器内 CK 源码路径为 `/home/yiding12/workspace/rocm-libraries/projects/composablekernel`。

3. **gfx1250 例外**：所有针对 gfx1250 的编译或运行命令，**不创建新容器**，改为 attach 到已有容器 `yiding12-gfx1250` 上执行：
   ```bash
   docker exec yiding12-gfx1250 bash -c "<命令>"
   ```

## Build 目录说明

`dev` preset 将 `binaryDir` 固定为 `${sourceDir}/build`。由于 gfx1250 容器（挂载到 `/root/workspace/...`）已占用 `build/`，gfx950 编译须用独立目录，通过 `-S`/`-B` 覆盖 preset：

```bash
cmake -S . -B build-gfx950 -GNinja --preset dev -DGPU_TARGETS='gfx950'
cmake --build build-gfx950 -j<num_cores> --target <target>
```

运行二进制时从对应 build 目录下执行（如 `cd build-gfx950 && bin/tile_example_fmha_bwd`）。

## gfx1250 构建说明

容器 `yiding12-gfx1250` 的挂载路径为 `-v /home/yiding12/workspace:/root/workspace`，因此容器内 CK 源码路径为：
```
/root/workspace/rocm-libraries/projects/composablekernel
```

## FMHA BWD 开发工作流

### 相关文件

- **入口**：`example/ck_tile/01_fmha/example_fmha_bwd.cpp`
- **Kernel 实现**：`example/ck_tile/01_fmha/fmha_bwd.hpp`
- **代码生成**：`example/ck_tile/01_fmha/generate.py` + `example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`
- **构建配置**：`example/ck_tile/01_fmha/CMakeLists.txt`
- **生成的 instance 文件**：`build/example/ck_tile/01_fmha/`（构建时生成，不在源码树中）

### 运行参数

参数格式为 `-key=value`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-v` | `1` | CPU 验证（**性能测试时设为 0**） |
| `-mode` | `0` | `0`:batch, `1`:group |
| `-b` | `2` | batch size |
| `-h` | `8` | Q 的 head 数 |
| `-h_k` | `-1` | K/V 的 head 数（-1 同 h，GQA/MQA） |
| `-s` | `3328` | seqlen_q |
| `-s_k` | `-1` | seqlen_k（-1 同 s） |
| `-d` | `128` | Q/K head dim |
| `-d_v` | `-1` | V head dim（-1 同 d） |
| `-prec` | `fp16` | 数据类型：`fp32`/`fp16`/`bf16` |
| `-mask` | `0` | `0`:无, `1`:top-left causal, `2`:bottom-right causal |
| `-kname` | `0` | 设为 1 可输出实际运行的 kernel instance 名称 |
| `-warmup` | `5` | 预热次数 |
| `-repeat` | `20` | 重复次数 |

```bash
# 性能测试示例（禁用验证）
./bin/tile_example_fmha_bwd -v=0 -b=2 -h=32 -s=4096 -d=128

# 查看实际运行的 kernel instance
./bin/tile_example_fmha_bwd -v=0 -b=2 -h=32 -s=16384 -d=64 -kname=1
```

### 快速编译（只编译目标 kernel instance）

BWD kernel 由三类 instance 组成，通过 `FMHA_BWD_FILTER` cmake 变量过滤，格式为：
```
dot_do_o模式@convert_dq模式@dq_dk_dv模式
```
使用 fnmatch 语法（`*` 通配符）。

**步骤一：用 `-kname=1` 找出目标场景的 kernel instance 名称**

```bash
./bin/tile_example_fmha_bwd -v=0 -kname=1 <其他参数>
# 输出示例（@分隔三个kernel）：
# fmha_bwd_dot_do_o_d64_fp16_b64_batch_o2_npad@fmha_bwd_convert_dq_d64_fp16_b64x0_batch_o2_npad_ndeterministic@fmha_bwd_d64_fp16_batch_..._ntrload
```

**步骤二：cmake configure 时传入 `FMHA_BWD_FILTER`**

```bash
# 针对 d=64, fp16, batch, 无 mask/bias/dropout 场景的示例：
cmake .. -GNinja --preset dev -DGPU_TARGETS='<arch>' -B build-<arch> \
  -DFMHA_BWD_FILTER='fmha_bwd_dot_do_o_d64_fp16_b64_batch_o2_npad@fmha_bwd_convert_dq_d64_fp16_b64x0_batch_o2_npad_ndeterministic@fmha_bwd_d64_fp16_batch_b32x128x64x32x64x32x32x64x64_r1x4x1_r4x1x1_r1x4x1_w16x16x32_w16x16x16_o1_maxq0_npad_nbias_ndbias_nmask_ndropout_ndeterministic_ntrload'
ninja -j$(nproc) tile_example_fmha_bwd
```

效果：编译单元从 ~1180 个降至 ~7 个，二进制从 ~90MB 降至 ~700KB。

**不传 `FMHA_BWD_FILTER` 时**编译全量 instances（适合最终验证或 CI）。
