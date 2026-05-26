# DistSpMV_Balanced

复现论文 **"Balancing Computation and Communication in Distributed Sparse Matrix-Vector Multiplication"** (CCGrid 2023)。

基于 MPI + OpenMP 并行实现了通信感知的分布式稀疏矩阵-向量乘法（算法 1–4）。

## 快速开始

### PowerShell（Windows）

```powershell
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载测试矩阵
#    前往 https://sparse.tamu.edu/ 搜索并下载所需 .mtx 文件，放入 data/ 目录
New-Item -ItemType Directory -Force -Path data
#    示例：下载 audikw_1 后解压移入
#    tar -xzf audikw_1.tar.gz -C data/
#    Copy-Item data/audikw_1/audikw_1.mtx data/

# 3. 使用 4 个 MPI 进程、每个进程 4 个线程运行（MS-MPI 无需 --oversubscribe）
mpiexec -n 4 python -m src.main `
    --matrix data/audikw_1.mtx `
    --threads 4 `
    --benchmark 50 `
    --output results/audikw_1_p4.json

# 4. 运行完整实验扫描
.\scripts\run_exp.ps1 -Suite representative -Procs 1,2,4,8 -Threads 4

# 5. 生成图表
python scripts/plot_results.py --results results/ --output figures/
```

### Bash（Linux / macOS）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载测试矩阵
#    前往 https://sparse.tamu.edu/ 搜索并下载所需 .mtx 文件，放入 data/ 目录
mkdir -p data
#    示例：下载 audikw_1 后解压移入
#    tar -xzf audikw_1.tar.gz -C data/
#    cp data/audikw_1/audikw_1.mtx data/

# 3. 使用 4 个 MPI 进程、每个进程 4 个线程运行
mpiexec -n 4 --oversubscribe python -m src.main \
    --matrix data/audikw_1.mtx \
    --threads 4 \
    --benchmark 50 \
    --output results/audikw_1_p4.json

# 4. 运行完整实验扫描
bash scripts/run_exp.sh --suite representative --procs 1,2,4,8 --threads 4

# 5. 生成图表
python scripts/plot_results.py --results results/ --output figures/
```

## 项目结构

```
DistSpMV_Balanced/
├── src/
│   ├── utils.py              # CSR 读写、性能指标、行划分
│   ├── reordering.py         # METIS / RCM 矩阵重排序
│   ├── partition.py          # 算法 1：对角块扩展
│   ├── comm_setup.py         # 算法 2：通信拓扑构建
│   ├── spmv_mpi_omp.py       # 算法 3+4：MPI 通信 + 多线程 SpMV
│   └── main.py               # 流水线编排、命令行接口、性能基准测试
├── scripts/
│   ├── run_exp.sh            # 一键实验启动脚本（Linux/macOS）
│   ├── run_exp.ps1           # 一键实验启动脚本（Windows PowerShell）
│   └── plot_results.py       # GFlops、加速比、热力图可视化
├── tests/
│   └── test_spmv.py          # 单元测试（pytest）
├── config.yaml               # 配置模板
├── requirements.txt          # Python 依赖
├── EXPERIMENT_REPORT.md      # 实验报告模板
└── README.md                 # 本文件
```

## 算法 ↔ 代码映射

| 论文       | 模块                  | 核心函数                   | 描述                      |
|------------|-----------------------|----------------------------|---------------------------|
| 算法 1     | `src/partition.py`    | `diagonal_block_expand()`  | 对角块列边界扩展           |
| 算法 2     | `src/comm_setup.py`   | `build_schedule()`         | 通信调度构建               |
| 算法 3     | `src/spmv_mpi_omp.py` | `_exchange_remote()`       | MPI Isend/Irecv 向量交换   |
| 算法 4     | `src/spmv_mpi_omp.py` | `_local_spmv()`            | 多线程本地 SpMV            |

### 算法 1 详解

**输入：** 本地拥有行的 CSR 数据，全局行范围 `[r_start, r_end)`。

**执行流程：**
1. 将对角块初始化为 `[r_start, r_end)`。
2. 统计 `diag_nnz[i]` = 第 `i` 行中列索引落在 `[left, right)` 内的非零元素个数。
3. `lower_bound = MPI_Allreduce(MAX_i diag_nnz[i], MPI_MAX)`，上限为 `N`。
4. 贪心扩展：当存在任一行的 `diag_nnz[i] < min(lower_bound, total_nnz[i])` 时：
   - 统计有多少不足行在列 `left-1` 处有非零元素（左增益）。
   - 统计有多少不足行在列 `right` 处有非零元素（右增益）。
   - 将边界向增益较大的一侧扩展 1 列。
   - 更新受影响行的 `diag_nnz`。
5. 返回 `(left, right)`。

**数据结构：** `rowptr[nlocal+1]`、`colidx[nnz_local]`（均为 `int32`），`diag/off_left/off_right` 数组（`int32`，长度 `nlocal`）。

**边界条件：**
- `0 ≤ left ≤ r_start ≤ r_end ≤ right ≤ N`
- 当边界到达 `0` 或 `N`，或所有行均已满足时，扩展停止。

### 算法 2 详解

**输入：** 本地 CSR、算法 1 得出的边界、所有进程的行分布信息。

**执行流程：**
1. `MPI_Allgather` 收集所有进程的 `(left, right, r_start, r_end)`。
2. 扫描非对角非零元素；对每个列 `c`，找到其所属的进程 `q`（满足 `r_start_q ≤ c < r_end_q`）。
3. 构建 `recvid[q]` = 需要从进程 `q` 获取的去重列集合。
4. `MPI_Alltoall` 交换接收数量 → 由此确定发送数量。
5. `MPI_Isend/Irecv` 交换列索引列表。
6. 从接收到的列表构建 `sendid[q]`。

**去重：** 使用 Python `set`，再转换为有序 `np.ndarray`。

### 算法 3 详解

**通信模式（每次 SpMV 调用）：**
1. 将本地 `x[left:right]` 复制到 `x_buf[0:diag_len)`。
2. 对每个对等进程 `q`：打包 `x_global[sendid[q][k]]` → 发送缓冲区。
3. 发送 `MPI_Irecv` 接收传入的向量元素。
4. 发送 `MPI_Isend` 将打包好的缓冲区发出。
5. `MPI_Waitall` 后将接收数据解包到 `x_buf`。

**标签约定：** 接收用 `tag = 200 + source_rank`，发送用 `tag = 200 + my_rank`。

### 算法 4 详解

**负载均衡：** 将行按非零元素数量划分，使每个线程处理约 `total_nnz / n_threads` 个非零元素。

**避免竞争：** 每个线程将结果累加到**私有的** `y` 数组；最终归约将所有私有数组求和。

## 环境配置

### MPI 安装

**Linux（Ubuntu/Debian）：**
```bash
sudo apt install openmpi-bin libopenmpi-dev
pip install mpi4py
```

**macOS：**
```bash
brew install open-mpi
pip install mpi4py
```

**Windows：**
1. 从 [Microsoft 官网](https://www.microsoft.com/en-us/download/details.aspx?id=100593) 下载安装 MS-MPI（`msmpisetup.exe` + `msmpisdk.msi`）
2. 将 `C:\Program Files\Microsoft MPI\Bin` 添加到系统 PATH（控制面板 → 系统 → 高级系统设置 → 环境变量）
3. 重启终端，安装 mpi4py：
```bash
pip install mpi4py
```

### METIS（可选）

用于 `--reorder metis` 选项：

**Linux：**
```bash
sudo apt install libmetis-dev
pip install pymetis
```

**不使用 METIS 时：** 默认的 `--reorder rcm` 使用 SciPy 的 Reverse Cuthill-McKee 算法（无需外部依赖）。

## 命令行参考

```
python -m src.main [OPTIONS]

选项：
  --matrix PATH      矩阵文件路径（.mtx，必填）
  --threads N        每个进程用于本地 SpMV 的线程数（默认：1）
  --reorder METHOD   重排序方法：rcm | metis | none（默认：rcm）
  --benchmark N      计时 SpMV 重复次数（默认：50）
  --warmup N         预热调用次数（默认：5）
  --seed N           随机种子（默认：42）
  --output PATH      JSON 输出文件路径（可选）
  --no-verify        跳过 scipy 正确性校验
  --verbose          输出 DEBUG 级别日志
```

## 运行测试

**PowerShell（Windows）：**
```powershell
# 单元测试（无需 MPI）
python -m pytest tests/test_spmv.py -v

# MPI 测试（需要 mpiexec）
mpiexec -n 1 python -m pytest tests/test_spmv.py -v -k "TestEndToEnd"
```

**Bash（Linux / macOS）：**
```bash
# 单元测试（无需 MPI）
python -m pytest tests/test_spmv.py -v

# MPI 测试（需要 mpiexec）
mpiexec -n 1 python -m pytest tests/test_spmv.py -v -k "TestEndToEnd"
```

## 已知限制

1. **GIL 限制的多线程：** Python 线程共享 GIL，限制了多线程 SpMV 的加速效果。要获得真正的 OpenMP 性能，需使用 C++ 或 Numba JIT 编译后端。
2. **无 GPU 支持：** 此实现仅针对 CPU 集群。
3. **仅支持对称重排序：** 列置换与行置换相同；非方阵需要分别指定行/列置换。
4. **METIS 回退：** 当 METIS 不可用时使用 RCM 替代；RCM 可减少带宽，但不会针对特定进程数进行优化。
5. **通信开销：** 在使用超过 16 个进程的实验中，对于小矩阵，MPI 延迟可能成为主要瓶颈。

## 参考文献

- 原始论文："Balancing Computation and Communication in Distributed Sparse Matrix-Vector Multiplication"，CCGrid 2023。
- SuiteSparse 矩阵集合：https://sparse.tamu.edu/
- mpi4py：https://mpi4py.readthedocs.io/
