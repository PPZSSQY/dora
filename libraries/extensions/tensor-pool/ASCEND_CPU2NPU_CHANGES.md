# 昇腾 CPU→NPU 设备 IPC 改动记录

## 范围

本次只实现单机 `CPU → NPU` tensor-pool，使用 `torch_npu` 的 Storage IPC。

不包含：

- NPU→NPU
- 跨机 Ascend 传输
- 裸 `aclrtIpcMemGetExportKey` / `aclrtIpcMemImportByKey` 路径
- README 修改
- 环境安装和 NPU 实机测试

## 传输流程

1. 发送进程根据接收设备创建 `torch.empty(..., device="npu:N")`。
2. 通过 `aclrtMemcpy(..., ACL_MEMCPY_HOST_TO_DEVICE)` 把 CPU 首帧拷入 NPU tensor。
3. 调用 `tensor.untyped_storage()._share_npu_()` 导出 torch_npu IPC 元组。
4. 将元组中的 bytes 字段编码为 Base64，并写入 dora `/dev/shm` 段的 JSON 元数据。
5. 固定头 `ipc_flag` 写为 `2`；CUDA IPC 继续使用原值 `1` 和原 64 字节槽。
6. 接收进程读取 JSON，调用 `torch.UntypedStorage._new_shared_npu(...)` 导入同一块 NPU 显存，并重建 tensor。
7. 后续 `write_tensor_pool` 只把新 CPU 帧 H2D 覆盖到发送端缓存的同一块 NPU tensor；接收端继续使用已导入的同一块显存。

## 改动文件

### `python/src/transport.rs`

#### IPC 类型

原逻辑：

- `ipc_flag == 0`：无设备 IPC。
- `ipc_flag == 1`：CUDA IPC，64 字节 handle 位于固定头 `[32..96)`。

现逻辑：

- `IPC_NONE = 0`
- `IPC_CUDA = 1`
- `IPC_ASCEND = 2`

Ascend 不占用 CUDA 的 64 字节槽；torch_npu IPC 元组放在 JSON 的 `npu_ipc` 字段中。

#### Ascend Python helper

在原 `get_ascend_helpers` 内增加：

- `_create_npu_ipc_buffer`：创建 NPU tensor、首帧 H2D、调用 `_share_npu_` 并返回 JSON 可序列化元数据。
- `_npu_memcpy_htod`：后续帧原地 H2D。
- `_import_npu_ipc_tensor`：调用 `_new_shared_npu` 导入 Storage，并按 dtype/shape 重建 tensor。
- `_release_npu_sender` / `_release_npu_receiver`：释放两侧 Python 缓存，使 torch_npu Storage 按生命周期析构。

原来的 `_npu_memcpy_dtoh` 和 NPU→CPU 路径保持不变。

#### `register_tensor_pool`

原逻辑：

- 所有 NPU 接收端都会直接报错。
- 同机 CUDA 接收端创建 header-only shmem；NPU 接收端没有对应分支。

现逻辑：

- 允许单机 CPU 源注册到 `npu` / `npu:N`。
- 明确拒绝 NPU→NPU 和跨机 NPU 接收端。
- CPU→NPU 创建 header + JSON 的小型 shmem，不创建 payload 数据区。
- 首帧在注册阶段完成 H2D 和 torch_npu IPC 导出。
- `pinned_type` 写为 `npu`，`device` 保存完整目标设备名。

#### `write_tensor_pool`

原逻辑：

- `ipc_flag == 1` 全部按 CUDA IPC 处理。
- CPU→NPU 不存在。

现逻辑：

- `ipc_flag == 2` 时，只接受 CPU 源。
- seqlock 置奇数后调用 `_npu_memcpy_htod`，成功后发布偶数 generation。
- H2D 失败时保持奇数，接收端不会消费半帧。
- CUDA 分支仍只处理 `ipc_flag == 1`。

#### `read_tensor_pool`

原逻辑：

- 设备 IPC 仅能导入 CUDA 64 字节 handle，并返回 CUDA 裸指针。

现逻辑：

- `ipc_flag == 2` 时解析 JSON 中的 `npu_ipc`。
- 调用 `_new_shared_npu` 导入 NPU Storage，重建并缓存 NPU tensor。
- 返回的 `tensor_info` 增加内部 `_tensor` 对象，并保留 ptr/size/dtype/shape/device。
- 导入前校验 daemon 可信 size、shape、dtype 和 IPC storage capacity。

#### 生命周期

原逻辑：

- free / 远端 drop 只清理 CPU shmem 和 CUDA 缓存。

现逻辑：

- 接收端 free 时删除导入的 NPU tensor 缓存。
- 发送端收到 pool drop 或主动 free 时删除导出的 NPU tensor 缓存。
- 依赖 torch_npu Storage 对象析构完成 IPC 引用与映射释放。

### `python/dora_tensor_pool/tensor_info.py`

原逻辑：

- 只支持 CPU 裸指针和 CUDA `__cuda_array_interface__`。

现逻辑：

- 若 transport 返回 `_tensor`，直接返回已由 torch_npu IPC 重建的 NPU tensor。
- CPU / CUDA 原有逻辑不变。

### `examples/receiver.py`

原逻辑：

- 只初始化 CUDA 接收设备。

现逻辑：

- NPU 接收端导入 `torch_npu` 并调用 `torch.npu.set_device`。

### `examples/cpu2npu.yml`

新增单机 CPU→NPU 吞吐示例：

- `sender_device: cpu`
- `receiver_device: npu`
- 复用 `sender.py`、`receiver.py` 和逐帧握手机制

## 验证

本地仅执行了不依赖 NPU 环境的静态检查：

```text
cargo fmt --all
cargo check -p dora-tensor-pool-python
```

NPU 实机测试由用户在 CANN 9.0 / 910C 环境执行：

```bash
dora run libraries/extensions/tensor-pool/examples/cpu2npu.yml
```

期望接收端输出 `device='npu:0'` 的相同数据，并在 100 帧后正常释放 pool。
