# 昇腾 NPU→NPU 设备 IPC 改动记录

## 范围

单机 `NPU → NPU` tensor-pool，对齐现有 CUDA→CUDA：

- 同卡：`npu:0 → npu:0`（示例 `examples/npu_inner.yml`）
- 跨卡：`npu:0 → npu:1`（示例 `examples/npu2npu.yml`）。能 P2P 则设备直拷，否则 CPU 中转

不包含：

- 跨机昇腾
- NPU→CUDA / CUDA→NPU
- README 修改
- 环境安装和 NPU 实机测试

## 传输流程

1. 发送进程在**接收卡**上 `aclrtMalloc` 池缓冲（跨卡用 `ACL_MEM_MALLOC_HUGE_FIRST_P2P`）。
2. 把源 NPU 帧拷进池：同卡 `aclrtMemcpy` D2D；跨卡先 `aclrtDeviceCanAccessPeer`，能则 `EnablePeerAccess` + D2D，不能则 `aclrtMallocHost` 做 D2H→H2D。
3. `aclrtIpcMemGetExportKey` 写出 65 字节 key，写入固定头 handle 槽，`device_type=2`。
4. 接收进程 `aclrtIpcMemImportByKey`（同卡导入，`IMPORT_FLAG_DEFAULT`），`tensor_from_info` 用 `torch.from_dlpack` 包装。
5. 后续 `write_tensor_pool` 按注册时选定的路径覆盖同一块池缓冲。

## 改动文件

### `python/src/transport.rs`

#### 注册准入

原逻辑：

- 接收端是 NPU 时只允许 CPU 源。
- NPU 源只允许 CPU 接收端。
- 跨机 NPU 接收端拒绝。

现逻辑：

- 允许单机 `cpu→npu` 与 `npu→npu`。
- NPU 源到非 CPU / 非 NPU 仍拒绝。
- 跨机 NPU 接收端仍拒绝。

#### Ascend helper

原逻辑：

- `_create_npu_ipc_buffer` 只做 CPU→NPU 的 H2D + 导出。
- `_npu_memcpy_htod` 只接受 CPU 源。

现逻辑（在原 helper 上增加，不改 CUDA helper）：

- `_setup_npu_ipc_from_npu`：接收卡建池、首帧 D2D / P2P / 中转、导出 65 字节 key。
- `_npu_memcpy_npu_to_pool`：后续帧按同一路径覆盖。
- `_npu_can_access_peer` / `_npu_ensure_p2p_pair`：对应 CANN 的 peer 查询与打开。
- `_release_npu_sender`：额外 `aclrtFreeHost` 释放中转缓冲。

CPU→NPU 的 `_create_npu_ipc_buffer` / `_npu_memcpy_htod` 不变。

#### `write_tensor_pool`

原逻辑：

- `device_type==2` 只接受 CPU 源。
- NPU 源遇到任何非 CPU 池（含昇腾 IPC）会报错。

现逻辑：

- `device_type==2` 且源是 NPU：走 `_npu_memcpy_npu_to_pool`。
- `device_type==2` 且源是 CPU：仍走 H2D。
- NPU 源只拒绝写入 **CUDA** IPC 池。

快路径、慢路径同样改。

### `examples/npu2npu.yml`

新建，对齐 `cuda2cuda.yml`：`npu:0 → npu:1`。

### `examples/npu_inner.yml`

新建，对齐 `cuda_inner.yml`：`npu:0 → npu:0`。

### `examples/npu2npu_nop2p.yml`

新建：跨卡但 `DORA_TENSOR_POOL_ASCEND_DISABLE_P2P=1`，必须走 host transit。

复用现有 `sender.py` / `receiver.py`（已有 NPU 初始化）。

## 验证

本地仅做不依赖 NPU 的静态检查（fmt / clippy / 相关单测）。实机由用户执行：

```bash
dora run libraries/extensions/tensor-pool/examples/npu_inner.yml
dora run libraries/extensions/tensor-pool/examples/npu2npu.yml
```

期望接收端 `device='npu:0'`（同卡）或 `device='npu:1'`（跨卡），100 帧后正常释放。

注册时发送端会打印一行路径，例如：

```text
Ascend NPU IPC path: same-device D2D
Ascend NPU IPC path: P2P D2D
Ascend NPU IPC path: host transit
```

强制不走 P2P（即使驱动支持 peer）时设 `DORA_TENSOR_POOL_ASCEND_DISABLE_P2P=1`，或跑：

```bash
dora run libraries/extensions/tensor-pool/examples/npu2npu_nop2p.yml
```

该示例必须打印 `Ascend NPU IPC path: host transit`。单元测试 `npu_cross_card_disable_p2p_must_use_host_transit` 锁住同一条决策。
