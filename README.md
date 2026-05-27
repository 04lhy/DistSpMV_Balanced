# DistSpMV_Balanced (C++)

复现论文 **"Balancing Computation and Communication in Distributed Sparse Matrix-Vector Multiplication"** (CCGrid 2023)。

基于 **MPI + OpenMP** 的 C++17 高性能实现，包含全部四个算法（Algorithm 1–4）。

## 为什么用 C++

Python 版本的 GIL 导致多线程本地 SpMV 无法真正并行执行。C++ 版本使用 OpenMP 实现真正的共享内存并行，消除分支预测开销，并直接使用原生 MPI 接口。对于 audikw_1 (77M 非零元) 级别的矩阵，实测可达 200x+ 性能提升（11.50 GFlops vs Python 的 0.051 GFlops）。

## 快速开始

### 1. 依赖安装

**编译器 & 工具链：**
- C++17 编译器 (GCC 9+, Clang 10+, MSVC 2019+, MinGW-w64 8+)
- MPI (OpenMPI / MPICH / MS-MPI)
- OpenMP (通常随编译器附带)
- CMake 3.16+
- Python 3.8+ (仅用于绘图脚本)

**Linux (Ubuntu/Debian):**
```bash
sudo apt install g++ cmake openmpi-bin libopenmpi-dev libmetis-dev
```

**macOS:**
```bash
brew install gcc cmake open-mpi metis
```

**Windows — MS-MPI：**
1. 从 [Microsoft 官网](https://www.microsoft.com/en-us/download/details.aspx?id=100593) 下载安装 MS-MPI SDK
2. 安装程序会自动将 `C:\Program Files\Microsoft MPI\Bin` 添加到 PATH

**Windows — MinGW-w64 (推荐，无需 Visual Studio)：**
```bash
# 通过 MSYS2 安装
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-openmp
# MS-MPI 仍需单独安装（见上）
```

### 2. 构建

项目现已支持 MinGW + MS-MPI 编译，CMakeLists.txt 会自动检测 MS-MPI 路径。

```bash
# ========== Linux / macOS ==========
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# ========== Windows (MinGW + MS-MPI) ==========
mkdir build && cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
mingw32-make -j$(nproc)

# ========== 带 METIS 支持 (Linux/macOS) ==========
# 如果已通过 apt/brew 安装 METIS，通常自动检测：
cmake .. -DCMAKE_BUILD_TYPE=Release

# ========== 带 METIS 支持 (Windows MinGW, 手动编译的 METIS) ==========
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release \
    -DMETIS_INCLUDE_DIR=/d/Tools/metis/include \
    -DMETIS_LIBRARY=/d/Tools/metis/lib/libmetis.a \
    -DGKLIB_LIBRARY=/d/Tools/gklib/lib/libGKlib.a
```

> **注意：** 如果不指定 METIS 变量，项目会以 `--reorder metis` 不可用的状态编译（仍可使用 `rcm` 和 `none`）。
> 不带 METIS 时 `find_package(METIS)` 失败仅打印一条 STATUS 信息，不影响构建。

### 3. 下载测试矩阵

从 [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) 下载 `.mtx` 文件，放入 `data/` 目录。

项目已自带：
- `data/test_small.mtx` — 10×10 三对角小矩阵，用于快速冒烟测试
- `data/audikw_1.mtx` — 94 万维, 7765 万非零元, 1.17 GB

### 4. 运行

```bash
# 冒烟测试 (10×10 小矩阵, 快速验证)
mpiexec -n 2 ./build/dist_spmv \
    --matrix data/test_small.mtx \
    --threads 2 --benchmark 10 --verbose

# 单机 4 进程, 每进程 4 线程, METIS 重排序
mpiexec -n 4 ./build/dist_spmv \
    --matrix data/audikw_1.mtx \
    --threads 4 --reorder metis \
    --benchmark 50 \
    --output results/audikw_1_p4.json

# 完整实验扫描 (Linux/macOS)
bash scripts/run_exp.sh --suite representative --procs 1,2,4,8 --threads 4

# 完整实验扫描 (Windows PowerShell)
.\scripts\run_exp.ps1 -Suite representative -Procs "1,2,4,8" -Threads 4

# 生成图表
python scripts/plot_results.py --results results/ --output figures/
```

## 项目结构

```
DistSpMV_Balanced/
├── CMakeLists.txt              # CMake 构建系统
├── src/
│   ├── types.hpp               # 公共数据结构 (CSRMatrix, CommSchedule)
│   ├── mmio.hpp / mmio.cpp     # Matrix Market (.mtx) I/O
│   ├── utils.hpp / utils.cpp   # CSR 工具, GFlops, 误差计算
│   ├── reordering.hpp/cpp      # RCM / METIS 矩阵重排序
│   ├── partition.hpp/cpp       # 算法 1: 对角块列边界扩展
│   ├── redistribute.hpp/cpp    # 远程矩阵按 nnz 重分布（论文第三节 B 部分）
│   ├── comm_setup.hpp/cpp      # 算法 2: 通信调度构建
│   ├── spmv_solver.hpp/cpp     # 算法 3+4: MPI 通信 + OpenMP 本地 SpMV
│   └── main.cpp                # 主入口, 流水线编排
├── scripts/
│   ├── run_exp.sh              # 实验启动脚本 (Linux/macOS)
│   ├── run_exp.ps1             # 实验启动脚本 (Windows)
│   └── plot_results.py         # 可视化 (GFlops, 加速比, 热力图)
├── tests/
│   └── test_spmv.cpp           # 单元测试 (C++)
├── config.yaml                 # 配置模板
└── README.md
```

## 算法 ↔ 代码映射

| 论文    | 源文件                | 核心函数                    | 描述                          |
|---------|-----------------------|-----------------------------|-------------------------------|
| 算法 1  | `src/partition.cpp`      | `diagonal_block_expand()`      | 贪心对角块列边界扩展           |
| nnz 重分布 | `src/redistribute.cpp` | `redistribute_remote_by_nnz()` | 按非零元数量重平衡远程矩阵，再运行算法 1 |
| 算法 2  | `src/comm_setup.cpp`     | `build_schedule()`             | Allgather → 扫描 → Alltoall → Isend/Irecv |
| 算法 3  | `src/spmv_solver.cpp` | `exchange_remote()`         | MPI 非阻塞向量交换             |
| 算法 4  | `src/spmv_solver.cpp` | `local_spmv()`              | OpenMP nnz-均衡本地 SpMV       |

## 算法详解

### 算法 1 — 对角块列边界扩展

**目标：** 确定每进程的对角块列范围 `[left, right)`，使每行至少获得 `lower_bound` 个对角块非零元，从而平衡计算量。

**步骤：**
1. **初始化** `left = r_start`, `right = min(r_end, N)`
2. 统计每行的 `diag_nnz[i]` — 列落在 `[left, right)` 内的非零元数量
3. `lower_bound = MPI_Allreduce(max(diag_nnz[i]), MPI_MAX)`
4. **贪心扩展：** 每次向能让更多「不足行」受益的方向扩展一列
   - 不足行：`diag_nnz[i] < min(lower_bound, total_nnz[i])`
   - 使用 `col_to_rows` 映射实现 O(不足行数) 的增益计算
5. 当所有行满足或无法继续扩展时停止

**复杂度：** O(N + nnz_local)，使用哈希表加速列→行查找。

### 远程矩阵 nnz 重分布（论文核心策略）

**目标：** 算法 1 确保了对角块内的计算均衡，但远程矩阵（`[left, right)` 之外的非零元）在各进程间可能严重不均——稀疏矩阵各行 nnz 差异可达几个数量级。论文的第三个关键策略是按非零元数量重新分配远程矩阵的行，使各进程负责的远程 nnz 总量大致相等。

**步骤：**
1. 统计每行在 `[left, right)` 外的远程非零元数量 `remote_nnz[i]`
2. `MPI_Allgather` 收集各进程的远程 nnz 总量，计算全局前缀
3. 根据 `global_remote_total / nprocs` 计算每进程的目标远程 nnz
4. 在 `q × target_per_proc` 边界处找到分割行（跨进程协调）
5. 通过 `MPI_Alltoall` + `MPI_Isend/Irecv` 交换行数据到新归属进程
6. 重排并重建本地 CSR；**重新运行算法 1** 以适配新的行范围

**复杂度：** O(nnz_local)，通信量与远程非零元规模成比例。

### 算法 2 — 通信调度构建

**目标：** 为每对进程构建 `sendid[q]`（要发送给 q 的列索引）和 `recvid[q]`（要从 q 接收的列索引）。

**步骤：**
1. `MPI_Allgather` 收集所有进程的 `(left, right, r_start, r_end)`
2. 扫描本地非零元，将非对角块列分配给其所属进程 → 构建 `recvid` 集合（去重）
3. `MPI_Alltoall` 交换收发数量
4. `MPI_Isend/Irecv` 交换实际的列索引列表
5. 从接收到的列表构建 `sendid`

**列归属：** 列 c 归属于 r_start_q ≤ c < r_end_q 的进程 q（按初始行分布）。

### 算法 3 — MPI 通信交换

每次 SpMV 调用执行：

1. 将本地 `x[left:right]` 复制到 `x_buf[0:diag_len)`
2. 对每个对端 q：打包 `x_global[sendid[q][k]]` → 发送缓冲区
3. 投递 `MPI_Irecv` 接收远程向量元素
4. 投递 `MPI_Isend` 发送打包的缓冲区
5. `MPI_Waitall` → 将接收数据解包到 `x_buf`

**标签约定：** 接收用 `tag = 200 + source_rank`，发送用 `tag = 200 + my_rank`

### 算法 4 — 多线程本地 SpMV

**统一 x_buf 布局：**
```
x_buf[0 .. diag_len)             — 本地的 x 元素
x_buf[diag_len .. diag_len+R)    — 来自对端的远程 x 元素
```

**`local_pos` 数组：** 每个非零元在 `x_buf` 中的位置，预处理阶段计算。热循环中无分支：
```cpp
for (j = rowptr[i]; j < rowptr[i+1]; j++)
    acc += val[j] * x_buf[local_pos[j]];
```

**负载均衡：** 行按非零元数量划分给线程，每个线程处理约 `total_nnz / n_threads` 个非零元。每个线程写入私有的 y 范围，无需锁或原子操作。

## 命令行参考

```
dist_spmv --matrix PATH [OPTIONS]

选项:
  --matrix PATH      矩阵文件路径 (.mtx, 必填)
  --threads N        每个进程的 OpenMP 线程数 (默认: 1)
  --reorder METHOD   重排序方法: rcm | metis | none (默认: rcm)
  --benchmark N      计时 SpMV 重复次数 (默认: 50)
  --warmup N         预热调用次数 (默认: 5)
  --seed N           随机种子 (默认: 42)
  --output PATH      JSON 输出文件路径 (可选)
  --no-verify        跳过正确性校验
  --verbose          启用详细日志输出
  --help             显示帮助信息
```

## 构建选项

| 选项 | 描述 |
|------|------|
| `-DCMAKE_BUILD_TYPE=Release` | 优化构建 (推荐) |
| `-DBUILD_TESTS=ON` | 构建测试可执行文件 |
| `-DMETIS_INCLUDE_DIR=<path>` | METIS 头文件路径 (`metis.h` 所在目录) |
| `-DMETIS_LIBRARY=<path>` | METIS 静态库路径 (`libmetis.a`) |
| `-DGKLIB_LIBRARY=<path>` | GKlib 静态库路径 (`libGKlib.a`)，METIS 依赖 |

> 三个 METIS 变量需同时指定。若使用 apt/brew 安装的 METIS 则通常无需手动指定，CMake 会自动查找。

## 运行测试

```bash
# 构建并运行
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make -j
mpiexec -n 1 ./test_spmv
```

## 环境要求

### MPI 安装

**Linux (Ubuntu/Debian):**
```bash
sudo apt install openmpi-bin libopenmpi-dev
```

**macOS:**
```bash
brew install open-mpi
```

**Windows:**
1. 从 [Microsoft 官网](https://www.microsoft.com/en-us/download/details.aspx?id=100593) 下载安装 MS-MPI
2. 安装程序会将 `C:\Program Files\Microsoft MPI\Bin` 添加到 PATH
3. CMakeLists.txt 已自动配置 MS-MPI 的 include/lib 路径

### METIS (可选, 用于 `--reorder metis`)

METIS 用于图划分重排序，能更好地将非零元聚集到对角块，减少通信量。

**Linux:**
```bash
sudo apt install libmetis-dev
```

**macOS:**
```bash
brew install metis
```

**Windows (MinGW, 从源码编译):**

METIS 依赖 GKlib（同作者的工具库），需先编译 GKlib 再编译 METIS。

```bash
# 1. 克隆源码
git clone https://github.com/KarypisLab/GKlib.git /tmp/gklib-src
git clone https://github.com/KarypisLab/METIS.git /tmp/metis-src

# 2. 编译安装 GKlib → /d/Tools/gklib
cmake -B /tmp/gklib-src/build -S /tmp/gklib-src \
    -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/d/Tools/gklib
cmake --build /tmp/gklib-src/build -j$(nproc)
cmake --install /tmp/gklib-src/build

# 3. 编译安装 METIS → /d/Tools/metis
# 先准备 build/xinclude 目录（METIS 构建系统的要求）
mkdir -p /tmp/metis-src/build/xinclude
echo '#define IDXTYPEWIDTH 32'  > /tmp/metis-src/build/xinclude/metis.h
echo '#define REALTYPEWIDTH 32' >> /tmp/metis-src/build/xinclude/metis.h
cat /tmp/metis-src/include/metis.h >> /tmp/metis-src/build/xinclude/metis.h
cp /tmp/metis-src/include/CMakeLists.txt /tmp/metis-src/build/xinclude/

cmake -B /tmp/metis-src/build -S /tmp/metis-src \
    -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/d/Tools/metis \
    -DGKLIB_PATH=/d/Tools/gklib \
    -DCMAKE_C_FLAGS="-D__USE_MINGW_ANSI_STDIO=1"
cmake --build /tmp/metis-src/build -j$(nproc)
# 注意：gpmetis/ndmetis/mpmetis 等命令行工具在 MinGW 下会编译失败，
# 但核心库 libmetis.a 已成功构建，项目只依赖该库。
cmake --install /tmp/metis-src/build || \
    (mkdir -p /d/Tools/metis/lib /d/Tools/metis/include && \
     cp /tmp/metis-src/build/libmetis/libmetis.a /d/Tools/metis/lib/ && \
     cp /tmp/metis-src/include/metis.h /d/Tools/metis/include/)

# 4. 构建 DistSpMV 时指定 METIS/GKlib 路径
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release \
    -DMETIS_INCLUDE_DIR=/d/Tools/metis/include \
    -DMETIS_LIBRARY=/d/Tools/metis/lib/libmetis.a \
    -DGKLIB_LIBRARY=/d/Tools/gklib/lib/libGKlib.a
cmake --build build -j$(nproc)
```

> **注意：** GKlib CMakeLists.txt 需要手动添加 `include/win32` 到 include 路径，
> 或将 `C:\Users\<user>\AppData\Local\Temp\gklib-src\CMakeLists.txt` 中
> `target_include_directories` 的 `$CMAKE_SOURCE_DIR/include` 后追加一行
> `$<$<PLATFORM_ID:Windows>:$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/win32>>`。

不使用 METIS 时，`--reorder rcm` 使用内置的 Reverse Cuthill-McKee 实现（零外部依赖），`--reorder none` 不做重排序。

## 性能参考

以下为 audikw_1 (94 万维, 7765 万非零元) 在本机 (MinGW + MS-MPI + OpenMP) 上的实测数据：

| 配置 | 重排序 | GFlops | 单次 SpMV (ms) | 通信量 (off-diag nnz) |
|------|--------|--------|----------------|----------------------|
| P=1, T=1 | none | 2.28 | 68.1 | 0 |
| P=4, T=4 | metis | **11.50** | 13.5 | ~199K (0.26%) |

Python (mpi4py) 对照 (P=4, T=4): 0.051 GFlops，C++ 约 **225x** 加速。

*注：实际性能取决于硬件、MPI 实现和矩阵特征。以上在 Intel i7-12700H (14C/20T), 16GB RAM 上测得。*

## 参考文献

- 原始论文: "Balancing Computation and Communication in Distributed Sparse Matrix-Vector Multiplication", CCGrid 2023
- SuiteSparse Matrix Collection: https://sparse.tamu.edu/
