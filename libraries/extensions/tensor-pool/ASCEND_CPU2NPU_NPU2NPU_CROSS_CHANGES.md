# 昇腾 CPU→NPU / NPU→NPU 跨机改动记录

## 范围

对齐 NVIDIA 的 `cpu2cuda_cross.yml` / `cuda2cuda_cross.yml`：

- **CPU→NPU 跨机**：A 的 CPU → A 的 `/dev/shm` → TCP/zenoh → B 的镜像 → B 上 `aclrtMemcpy` H2D
- **NPU→NPU 跨机**：A 的 NPU → A 上 D2H 进 `/dev/shm` → 同上 → B 上 H2D

这不是跨机设备 IPC，也不是远端显存直用。句柄不能跨主机。

不包含：

- 同机 `cpu2npu` / `npu2npu` IPC（保持原样）
- CUDA 路径的行为改动（只把镜像头里的 `cuda:0` 也识别成 CUDA 接收端，与 daemon 实际写入一致）
- 环境安装和实机测试

## 改动文件

### `python/src/transport.rs`

#### 跨机守卫

**位置：** `register_tensor_pool` 开头

| | 内容 |
|---|---|
| **原逻辑** | `receiver_is_npu && cross_machine` 直接 `bail`，所有跨机 NPU 接收端被拒。 |
| **现逻辑** | 去掉这条守卫。跨机 NPU 接收端改走主机数据区；源仍必须是 CPU 或 NPU。 |

#### 同机才建 Ascend IPC

**位置：** 创建 65 字节 key、写 `device_type`、`ipc_written`、注册失败检查

| | 内容 |
|---|---|
| **原逻辑** | 只要接收端是 NPU 就 `aclrtMalloc` + 导出 IPC，`device_type=2`。跨机也会建——但前面已经被拒，走不到。 |
| **现逻辑** | `receiver_is_npu && !cross_machine` 才建 IPC。跨机：`device_type=0`，满尺寸 shmem，首帧 `memcpy` 或已有 `copy_npu_to_host`，再 `register_cross_machine` + `push_mirror_update`。 |

#### 接收端 H2D 中转（Ascend helper）

**位置：** `get_ascend_helpers` 内嵌 Python

| | 内容 |
|---|---|
| **原逻辑** | 只有发送端 IPC 池和接收端 `aclrtIpcMemImportByKey`。 |
| **现逻辑** | 新增独立表 `_npu_htod_buffers`：`_get_npu_htod_buf`（`aclrtMalloc`）、`_npu_memcpy_htod_raw`（H2D）、`_release_npu_htod`（`aclrtFree`，不 `IpcMemClose`）。不和 CUDA helper 混用。 |

#### 读路径

**位置：** `try_doradma_read`

| | 内容 |
|---|---|
| **原逻辑** | `pinned_type == "npu"` 才当 NPU。镜像头里 daemon 写的是接收端设备名 `npu:0`，对不上就会落到 CPU 读。`device_type=0` 且要上 NPU 时没有 H2D 分支。 |
| **现逻辑** | `is_npu_device(pinned_type)`，因此 `npu` / `npu:0` 都算。`ipc_present==2` 仍走 IPC。否则若是 NPU 接收端：用 `RECV_NPU_HTOD` 缓存本地 `aclrtMalloc`，每帧从镜像数据区 H2D。设备号优先 JSON 的 `device`，否则用 `pinned_type`（镜像头通常只有后者）。 |

CUDA 对照：`pinned_type` 为 `cuda` 或 `cuda:` 前缀时走原有 `RECV_GPU_HTOD`。daemon 镜像头写的是 `cuda:0`，原先只认 `"cuda"`，跨机 GPU 读会误走 CPU；这里按同一约定补上前缀判断，不改 CUDA 拷贝实现。

#### 释放

**位置：** `free_tensor_pool`、`process_pending_tensor_pool_frees`

| | 内容 |
|---|---|
| **原逻辑** | 只清 CUDA `RECV_GPU_HTOD` 和 NPU IPC `_npu_receiver_buffers`。 |
| **现逻辑** | 额外 `RECV_NPU_HTOD` + `_release_npu_htod`。IPC 释放路径不变。 |

#### 单测

新增 `mirror_device_strings_count_as_npu_pinned_type`。原有 `host_payload_pools_are_eligible_to_push_mirror` 仍覆盖 `device_type=0` 会 push 镜像。

### `examples/cpu2npu_cross.yml`（新建）

对照 `cpu2cuda_cross.yml`：`sender_device: cpu`，`receiver_device: npu:0`。A 可装 CPU 版 torch；B 不写 `build:`，沿用本机 `torch_npu`。

### `examples/npu2npu_cross.yml`（新建）

对照 `cuda2cuda_cross.yml`：两端 `npu:0`（两台机器各自的 0 号卡，不是单机的 `npu:0→npu:1`）。两端都不 `pip install` 通用 torch。

### `README.md` / `examples/README.md`

写上跨机 CPU→NPU、NPU→NPU，并说明走主机中转。

## 手工验证

两端 `DORA_MEMORY_POOL_CROSS_MACHINE=1`，带 `tensor-pool` 的 CLI/daemon，连接方式与 `cpu2cpu_cross` 相同。

```bash
# A = CPU，B = 昇腾
dora start --coordinator-addr <B_IP> --coordinator-port 6025 \
  libraries/extensions/tensor-pool/examples/cpu2npu_cross.yml --attach

# A、B 都有昇腾
dora start --coordinator-addr <B_IP> --coordinator-port 6025 \
  libraries/extensions/tensor-pool/examples/npu2npu_cross.yml --attach
```

期望：receiver preview 与 sender 一致，设备为 `npu:0`，100 帧后正常结束。本改动不安装环境。
