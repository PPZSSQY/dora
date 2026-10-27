# 昇腾 NPU→CPU 跨机改动记录

## 范围

本次只补 **跨机 `NPU → CPU`**，对齐现有 `cuda2cpu_cross.yml`：

- 发送端在机器 A 的 NPU 上出帧
- 接收端在机器 B 的 CPU 上读帧
- 线上海量仍是主机字节：A 上 D2H 进 `/dev/shm`，再走已有 CPU 镜像 / direct-TCP

不包含：

- 跨机 CPU→NPU / NPU→NPU（接收端是 NPU 仍拒绝）
- daemon 数据面、zenoh、协议改动
- CUDA 路径改动
- 环境安装和实机测试

## 传输流程

和 NVIDIA `cuda2cpu_cross` 同一条主机中转，只是 D2H 用 AscendCL：

1. A 上 `register_tensor_pool(..., device="cpu", machine="B")`。
2. `aclrtMemcpy` D2H 把首帧写入 A 的全尺寸 shmem（`device_type=0`）。
3. daemon 在 B 建镜像段，首帧 `push_mirror_update`。
4. 后续 `write_tensor_pool` 再 D2H 覆盖 A 的数据区，`should_push_mirror` 因 `ipc_present == DEVICE_CPU` 为真，把新帧推到 B。
5. B 的接收节点映射本地镜像，按 CPU tensor 读。

这不是「B 的 CPU 直接用 A 的 NPU 显存」。CUDA IPC / Ascend IPC 都不能跨主机。

## 为什么几乎不改 `transport.rs`

单机 `npu2cpu` 已经会把 NPU 源写进主机数据区。跨机守卫只拒绝 **接收端是 NPU**：

```text
if receiver_is_npu && cross_machine { bail!(...) }
```

`NPU 源 + CPU 接收 + machine=B` 不会进这个分支。注册时 `!receiver_is_accelerator || cross_machine` 会走 D2H；写入后走原有 `push_mirror_update`。daemon 侧搬字节，不识别厂商。

## 改动文件

### `examples/npu2cpu_cross.yml`（新建）

| | 内容 |
|---|---|
| **原逻辑** | 无此文件；最接近的是 `cuda2cpu_cross.yml`（`sender_device: cuda:0`）。 |
| **现逻辑** | 拓扑与 `cuda2cpu_cross.yml` 相同（A 发送、B 接收、`cross_machine: "B"`）。`sender_device: npu:0`。A 不写 `build:`（沿用本机已装的 `torch_npu`）；B 仍装 CPU 版 torch。复用 `sender.py` / `receiver.py`（已有 NPU 初始化与 `cross_machine`）。 |

### `python/src/transport.rs`

#### 跨机守卫注释

**位置：** `register_tensor_pool` 里 `receiver_is_npu && cross_machine` 之前

| | 内容 |
|---|---|
| **原逻辑** | 注释写「跨机仍不支持」，容易理解成所有昇腾跨机都被拒。 |
| **现逻辑** | 写明：跨机拒的是 NPU **接收端**；NPU→CPU 跨机走主机数据区 + 现有 mirror，允许。守卫条件未改。 |

#### 单元测试 `host_payload_pools_are_eligible_to_push_mirror`

| | 内容 |
|---|---|
| **原逻辑** | 只断言 CUDA/Ascend IPC 池（`device_type` 1/2）不 push 镜像。 |
| **现逻辑** | 补一条：`DEVICE_CPU` 的池（NPU→CPU / CUDA→CPU 跨机都是这种）可以 push。 |

### `README.md`

| | 内容 |
|---|---|
| **原逻辑** | 「Ascend 目前仅单机 NPU→CPU；CPU→NPU / NPU→NPU / IPC / 跨机未实现」（后几项已被后续单机工作超越）。 |
| **现逻辑** | 写明单机 NPU↔CPU / NPU↔NPU，以及跨机 NPU→CPU；跨机 NPU 接收端仍未实现。 |

### `examples/README.md`

| | 内容 |
|---|---|
| **原逻辑** | 文件列表和跨机小节只有 CPU / CUDA `*_cross`。 |
| **现逻辑** | 增加 `npu2cpu_cross.yml`：A 要有昇腾，B 只要 CPU。 |

## 手工验证

环境：A 已配置 CANN、`torch_npu`、`libascendcl.so`；B 只要 CPU 版 torch。两端 CLI/daemon 带 `--features tensor-pool`，且 `DORA_MEMORY_POOL_CROSS_MACHINE=1`。启动方式与 `examples/README.md` 里 `cpu2cpu_cross.yml` / `cuda2cpu_cross.yml` 相同。

```bash
dora build --coordinator-addr <rendezvous-ip> --coordinator-port 6025 \
  libraries/extensions/tensor-pool/examples/npu2cpu_cross.yml
dora start --coordinator-addr <rendezvous-ip> --coordinator-port 6025 \
  libraries/extensions/tensor-pool/examples/npu2cpu_cross.yml --attach
```

期望：

- sender preview 在 `npu:0`
- receiver preview 数值相同、在 CPU 上
- 100 帧后两端正常结束

本改动不安装环境。同机双 daemon 若 `direct=true`，接收端会直读 A 的 shmem，测不到网线镜像；真跨机请用两台机器。
