# 昇腾 NPU→CPU 迁移改动记录

## 范围

本次改动只实现单机 `NPU → CPU` 的 tensor-pool 路径，面向 CANN 8.x/9.x，以及由 `torch_npu` 创建的张量。

对应提交：`feat(tensor-pool): add single-machine Ascend NPU-to-CPU transport`。

---

## 改动文件一览

| 文件 | 改动性质 |
|---|---|
| `python/src/transport.rs` | **核心逻辑**：识别 NPU、AscendCL 拷贝、注册/写入分支 |
| `python/src/lib.rs` | 包头注释：CUDA → CUDA/Ascend |
| `python/Cargo.toml` | crate 描述文案 |
| `examples/sender.py` | NPU 设备初始化（`torch_npu` / `torch.npu.set_device`） |
| `examples/npu2cpu.yml` | **新建**：NPU→CPU 吞吐示例 dataflow |
| `examples/README.md` | 补充 `npu2cpu.yml` 说明 |
| `README.md` | 补充 Ascend 支持范围说明 |
| `ASCEND_NPU2CPU_CHANGES.md` | 本改动记录（本文档） |

未改动的重要边界：

- `python/dora_tensor_pool/cuda.py`：仍只服务 CUDA IPC / `__cuda_array_interface__`，NPU→CPU 读侧是 CPU，不需要改。
- `src/daemon/*`：跨机 mirror / 数据面未动；本路径是单机 shmem。
- CUDA 助手 `get_cuda_helpers` / `libcudart.so`：保持原样，与 Ascend 分轨。

---

## 逐文件：原逻辑 → 现逻辑

### 1. `python/src/transport.rs`（核心）

#### 1.1 新增 Ascend 助手缓存与设备判定

**位置：** 文件前部（约 `ASCEND_HELPERS` / `is_npu_device` / `npu_device_index`）

| | 内容 |
|---|---|
| **原逻辑** | 只有 `CUDA_HELPERS`；设备判定只有 `starts_with("cuda")`。`npu` / `npu:0` 一律当非 CUDA，落到 CPU 分支。 |
| **现逻辑** | 新增独立的 `ASCEND_HELPERS`（不与 CUDA 共用）。`is_npu_device` 识别 `npu` / `npu:<index>`；`npu_device_index` 解析卡号（`npu` → 0）。附带单元测试，避免与 `cpu` / `cuda` 混淆。 |

#### 1.2 新增 `get_ascend_helpers` + `copy_npu_to_host`

**位置：** `get_cuda_helpers` 之后

| | 内容 |
|---|---|
| **原逻辑** | 设备→主机拷贝全部走 `get_cuda_helpers` → `libcudart.so` → `cudaMemcpy(..., kind=2)`。没有 AscendCL 绑定。 |
| **现逻辑** | 懒加载 ctypes 绑定 `libascendcl.so`，对外暴露 `_npu_memcpy_dtoh`。`copy_npu_to_host` 是 Rust 侧统一入口。 |

内嵌 Python 助手的拷贝步骤：

1. `_ensure_npu_runtime(device_index)`：进程内首次调用 `aclInit(NULL)`；若已由 `torch_npu` 初始化，忽略 `ACL_ERROR_REPEAT_INITIALIZE`（100002）；卡号变化时调用 `aclrtSetDevice`。
2. `aclrtSynchronizeDevice`：等待未完成的 NPU 任务。
3. `aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST)`：同步 D2H 进 `/dev/shm`。
4. **不**调用 `aclFinalize`，避免拆掉 `torch_npu` 仍在使用的运行时。

#### 1.3 `register_tensor_pool`：设备判定与守卫

**位置：** `register_tensor_pool` 开头解析 `tensor_info` / `device` 处

| | 内容 |
|---|---|
| **原逻辑** | `is_cuda = tensor_device.starts_with("cuda")`；`receiver_is_cuda = device.starts_with("cuda")`。`npu` 源会被当成 CPU 源。`should_pin(is_cuda, size)` 会把 NPU 源当成可 pin 的主机指针（错误）。 |
| **现逻辑** | 增加 `is_npu`。若接收端是 NPU，或 NPU 源但接收端不是 `cpu`，直接 `bail` 并说明「当前只支持 NPU→CPU」。`should_pin(is_cuda \|\| is_npu, size)`：NPU 源与 CUDA 源一样**不**对源指针做 `cudaHostRegister`。注册时若源是 NPU，不对 shmem 走 CUDA 的 `_register_host` pin。 |

#### 1.4 `register_tensor_pool`：首帧拷贝进 shmem

**位置：** 写完 DORADMA 头之后、「把首帧数据拷进 shmem」分支（约步骤 7）

| | 内容 |
|---|---|
| **原逻辑** | `if is_cuda { _cuda_memcpy(..., kind=2) } else { memcpy }`。`device="npu:0"` 时 `is_cuda==false`，走 `memcpy`，把 **NPU 设备指针当主机指针拷**，非法，可能崩溃。 |
| **现逻辑** | 分支变为：`if is_npu { copy_npu_to_host(...) } else if is_cuda { _cuda_memcpy(...) } else { memcpy }`。NPU 源走 AscendCL D2H；CUDA 源仍走原 CUDA 路径；纯 CPU 源仍 `memcpy`。失败则 reclaim shmem 并报错。 |

#### 1.5 `write_tensor_pool`：快路径 / 慢路径

**位置：** 快路径（`pool_` + `PINNED_POOL`）与慢路径（daemon 元数据 reopen）中，原 `else if is_cuda` 之前

| | 内容 |
|---|---|
| **原逻辑** | 仅 `is_cuda`。NPU 源落进 `else` 的 `memcpy`，同样非法。 |
| **现逻辑** | 增加 `is_npu` 与 `auto_pin = should_pin(is_cuda \|\| is_npu, size)`。快/慢路径在写数据前：若 `is_npu && ipc_present == 1`，拒绝写入（IPC 池没有 shmem 数据区，且本路径不支持 Ascend IPC）。`else if is_npu`：seqlock 标记写入中 → `copy_npu_to_host` → 成功则 `seqlock_end` 发布；失败保持 gen 为奇数，读者不会吃到半帧。原 `is_cuda` 分支顺序与语义不变。 |

#### 1.6 读路径

| | 内容 |
|---|---|
| **原逻辑 / 现逻辑** | NPU→CPU 的接收端是 CPU：`try_doradma_read` 走 shmem 数据区 + 返回 `device: "cpu"`。**本迁移未改读路径**；改的是发送端如何把 NPU 数据合法写进 shmem。 |

---

### 2. `examples/sender.py`

| | 内容 |
|---|---|
| **原逻辑** | 仅当 `SENDER_DEVICE` 以 `cuda` 开头时 `torch.cuda.set_device`。NPU 设备名会直接 `torch.tensor(..., device="npu")`，若未 import `torch_npu` 会失败。 |
| **现逻辑** | 增加 `elif SENDER_DEVICE.startswith("npu")`：`import torch_npu`，再 `torch.npu.set_device(idx)`。注释说明：PyTorch 侧先初始化；传输层再 `aclInit` 时遇到重复初始化按官方约定忽略。 |

---

### 3. `examples/npu2cpu.yml`（新建）

| | 内容 |
|---|---|
| **原逻辑** | 无此文件；最接近的是 `cuda2cpu.yml`（`sender_device: cuda`）。 |
| **现逻辑** | `sender_device: npu`，`receiver_device: cpu`，`message_num: 100`，节点仍用 `sender.py` / `receiver.py`。用于单机 Ascend 吞吐验证。 |

---

### 4. 文档与元数据（轻量）

| 文件 | 原逻辑 | 现逻辑 |
|---|---|---|
| `README.md` | 只描述 pinned-host / CUDA | 增加：Ascend 目前仅单机 NPU→CPU；CPU→NPU / NPU→NPU / IPC / 跨机未实现 |
| `examples/README.md` | 示例列表无 NPU | 增加 `npu2cpu.yml` 条目及依赖说明 |
| `python/src/lib.rs` | 注释写 CUDA tensor-pool | 改为 CUDA / Ascend |
| `python/Cargo.toml` | description 仅 CUDA | description 含 Ascend |

---

## 行为对照（端到端）

### 原先：`device=npu:0` 的发送端会发生什么

1. `is_cuda == false` → 当成 CPU 源。  
2. 注册/写入用 `memcpy` 拷 NPU 设备指针 → **未定义行为 / 可能崩溃**。  
3. 或根本起不来（未装 / 未 import `torch_npu`）。

### 现在：单机 NPU→CPU

1. sender 初始化 `torch_npu` 并在 NPU 上建张量。  
2. `register_tensor_pool(..., device="cpu")`：建全尺寸 shmem，首帧 `aclrtSynchronizeDevice` + `aclrtMemcpy` D2H。  
3. 每帧 `write_tensor_pool`：同样 D2H 覆盖 shmem 数据区，seqlock 发布。  
4. receiver `read_tensor_pool`：从 shmem 零拷贝读 CPU 视图（原有路径）。  
5. CUDA 路径仍只加载 `libcudart.so`，互不影响。

### 明确拒绝的路径（避免静默走错）

- 接收端为 `npu` / `npu:N`  
- NPU 源且接收端不是 `cpu`（例如误写成 `cuda:0`）  
- NPU 源写入 `ipc_present == 1` 的 CUDA IPC 池  

---

## 运行时初始化

示例 sender 在创建张量前会 `import torch_npu` 并调用 `torch.npu.set_device`，从而为 PyTorch 侧完成 AscendCL 初始化。

传输层在第一次 NPU 拷贝时也会显式初始化：

1. 每个进程调用一次 `aclInit(NULL)`。  
2. 若 `torch_npu` 已经初始化过，返回 `ACL_ERROR_REPEAT_INITIALIZE`（100002）——按官方约定忽略并继续。  
3. 张量 NPU 卡号变化时调用 `aclrtSetDevice(index)`。  
4. 不调用 `aclFinalize`（避免拆掉 `torch_npu` 仍在用的运行时；进程退出再释放）。

---

## 有意暂不支持的范围

- CPU→NPU  
- NPU→NPU，以及昇腾跨进程 IPC  
- 跨机昇腾传输  
- 目标 `/dev/shm` 目前是可换页主机内存（第一版优先正确性；锁页优化可后续单独做）

---

## 手工验证

环境：已配置昇腾运行时、`torch_npu`、CANN 8.x/9.x、`libascendcl.so`；Python 包与 CLI 需带 `--features tensor-pool` 构建。

```bash
maturin develop -m apis/python/node/Cargo.toml --features tensor-pool
cargo build -p dora-cli --features tensor-pool
export PATH="$PWD/target/debug:$PATH"

dora run libraries/extensions/tensor-pool/examples/npu2cpu.yml
```

成功标志示例：

- sender 打印 `device='npu:0'` 的 preview  
- receiver 打印相同数据的 CPU preview  
- 打印吞吐并两节点 `finished successfully`  

本改动不会自动安装任何昇腾环境或依赖。



((.venv) ) root@70f501bb5124:/workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool# cargo build -p dora-cli --features tensor-pool
export PATH=/workspace/tensorpool-ascend/dora/target/debug:$PATH
dora run /workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/examples/npu2cpu.yml
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.43s
receiver_node: DEBUG    building node
sender_node: DEBUG    building node
03:52:48 DEBUG  receiver_node: daemon::spawner spawning node
03:52:48 DEBUG  sender_node: daemon::spawner spawning node
03:52:48 INFO   receiver_node: spawner spawning: "/workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/.venv/bin/python" -u /workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/examples/receiver.py
03:52:48 INFO   sender_node: spawner spawning: "/workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/.venv/bin/python" -u /workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/examples/sender.py

03:52:48 INFO   [dora]: dora daemon finished building nodes, spawning...

03:52:48 INFO   receiver_node: spawner spawning `/workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/.venv/bin/python` in `/workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/examples`
03:52:49 DEBUG  receiver_node: spawner spawned node with pid 3707689
03:52:49 INFO   sender_node: spawner spawning `/workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/.venv/bin/python` in `/workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/examples`
03:52:49 DEBUG  sender_node: spawner spawned node with pid 3707690
03:52:53 INFO   receiver_node: daemon node is ready
03:53:01 INFO   sender_node: daemon node is ready
03:53:01 INFO   [dora]: daemon all nodes are ready, starting dataflow
03:53:01 stdout sender_node: Sender preview: tensor([  0,  18, 791, 697, 792], device='npu:0')
03:53:01 stdout receiver_node: Receiver preview: tensor([  0,  18, 791, 697, 792])
100%|██████████| 100/100 [00:03<00:00, 30.62it/s]
03:53:04 stdout receiver_node: Average transfer throughput: 8146.654811 MB/s
03:53:05 DEBUG  receiver_node: daemon handling node stop with exit status Success (restart: false, restart_count: 0)
03:53:05 INFO   receiver_node: daemon receiver_node finished successfully
03:53:09 DEBUG  sender_node: daemon handling node stop with exit status Success (restart: false, restart_count: 0)
03:53:09 INFO   sender_node: daemon sender_node finished successfully
03:53:09 INFO   [dora]: daemon dataflow finished on machine `01a161bd-762f-7148-b161-59a7bb36c315`
2026-10-22T03:53:09.623844Z  INFO run_inner: dora_daemon: exiting daemon because all required dataflows are finished health_check_interval_duration=None self.daemon_id=DaemonId { machine_id: None, uuid: 01a161bd-762f-7148-b161-59a7bb36c315 }
