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

1. 发送进程 `aclrtMalloc`（按 4KB 对齐）分配 NPU 显存。
2. 通过 `aclrtMemcpy(..., ACL_MEMCPY_HOST_TO_DEVICE)` 把 CPU 首帧拷入该显存。
3. 调用 `aclrtIpcMemGetExportKey(..., 65, DISABLE_PID_VALIDATION)` 导出 65 字节 IPC key。
4. 把 key 写入固定头 handle 槽 `[32, 97)`，其余 7 字节填 0；`device_type=2`。
5. 接收进程从固定头读 65 字节，调用 `aclrtIpcMemImportByKey` 得到本进程 device 指针。
6. `tensor_from_info` 用 `torch.from_dlpack` 把该指针包成 NPU tensor（零拷贝）。
7. 后续 `write_tensor_pool` 只把新 CPU 帧 H2D 覆盖到发送端同一块 `aclrtMalloc` 显存。

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

## 后续：固定头统一与 ACL 65 字节 IPC

上面第一节是最初实现。后来为了和 CUDA 固定头对齐，做了如下替换（**只认新头，不兼容旧头**）。

### 新固定头

| 偏移 | 长度 | 字段 |
|------|------|------|
| 24 | 8 | `device_type`（原 `ipc_flag`，数字仍是 0=CPU / 1=NVIDIA / 2=昇腾） |
| 32 | 72 | handle 槽：CUDA 前 64 字节，昇腾前 65 字节，其余填 0 |
| 104 | 8 | `write_generation`（从 96 挪到这里，8 字节对齐） |
| 112 | 8 | `header_version = 1`（只作版本标记，读侧不回退旧偏移） |
| 256 | N | JSON（不再放 handle / `npu_ipc`） |

读写 generation **一律 104**。旧 `/dev/shm`、旧 daemon、旧节点不支持，必须整包重编。

### `python/src/transport.rs`

原逻辑：

- handle 槽只有 64 字节，`write_gen` 在 96。
- 昇腾走 `torch_npu` `_share_npu_` 八元组，放进 JSON `npu_ipc`。
- 字段名 `ipc_flag`。

现逻辑：

- `DEVICE_CPU/CUDA/ASCEND = 0/1/2`，头字段改称 `device_type`。
- `_create_npu_ipc_buffer`：`aclrtMalloc` + H2D + `aclrtIpcMemGetExportKey`，返回 65 字节 key。
- `_import_npu_ipc`：从固定头读 65 字节，`aclrtIpcMemImportByKey`，返回 device 指针。
- `_release_npu_sender`：`aclrtIpcMemClose` + `aclrtFree`。
- `_release_npu_receiver`：`aclrtIpcMemClose`（不 `aclrtFree`，映射不是本进程分配的）。
- CUDA 仍写 64 字节到槽前部，并显式把槽内剩余字节清零。

### `src/daemon/mirror.rs` / `data_plane.rs` / `mod.rs`

原逻辑：镜像段 `write_gen` 写在偏移 96。

现逻辑：与节点头一致，写/读偏移 **104**，并写 `header_version=1`。跨机镜像仍不导出设备 handle（`device_type=0`）。

### `python/dora_tensor_pool/tensor_info.py`

原逻辑：transport 返回 `_tensor`（`_new_shared_npu` 重建）。

现逻辑：transport 只返回 NPU 裸指针；`tensor_from_info` 用 `torch.from_dlpack` 零拷贝包装。包装出的 tensor 不拥有显存。

## 生命周期注意（SIGSEGV）

昇腾 IPC 仍要求：**消费者先释放导入映射，生产者才能 close/free**。

示例 `receiver.py` 在最后一帧 `free_tensor_pool` 之后、发送 `next_require` 之前，对 NPU 接收端清除 `torch_tensor` 和 `tensor_info`，再 `gc.collect()`，避免发送端先退出。

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

((.venv) ) root@70f501bb5124:/workspace/tensorpool-ascend/dora# cargo build -p dora-cli --features tensor-pool
export PATH=/workspace/tensorpool-ascend/dora/target/debug:$PATH
dora run /workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/examples/cpu2npu.yml
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.42s
receiver_node: DEBUG    building node
sender_node: DEBUG    building node
09:00:42 DEBUG  receiver_node: daemon::spawner spawning node
09:00:42 DEBUG  sender_node: daemon::spawner spawning node
09:00:42 INFO   receiver_node: spawner spawning: "/workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/.venv/bin/python" -u /workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/examples/receiver.py
09:00:42 INFO   sender_node: spawner spawning: "/workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/.venv/bin/python" -u /workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/examples/sender.py

09:00:42 INFO   [dora]: dora daemon finished building nodes, spawning...

09:00:42 INFO   receiver_node: spawner spawning `/workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/.venv/bin/python` in `/workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/examples`
09:00:42 DEBUG  receiver_node: spawner spawned node with pid 3849567
09:00:42 INFO   sender_node: spawner spawning `/workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/.venv/bin/python` in `/workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/examples`
09:00:42 DEBUG  sender_node: spawner spawned node with pid 3849568
09:00:47 INFO   sender_node: daemon node is ready
09:00:47 INFO   receiver_node: daemon node is ready
09:00:47 INFO   [dora]: daemon all nodes are ready, starting dataflow
09:00:47 stdout sender_node: Sender preview: tensor([  0, 847,  95, 158, 357])
09:00:48 stdout receiver_node: Receiver preview: tensor([  0, 847,  95, 158, 357], device='npu:0')
100%|██████████| 100/100 [00:03<00:00, 32.08it/s]
09:00:51 stdout receiver_node: Average transfer throughput: 19031.256417 MB/s
09:00:54 DEBUG  sender_node: daemon handling node stop with exit status Success (restart: false, restart_count: 0)
09:00:54 INFO   sender_node: daemon sender_node finished successfully
09:00:55 DEBUG  receiver_node: daemon handling node stop with exit status Success (restart: false, restart_count: 0)
09:00:55 INFO   receiver_node: daemon receiver_node finished successfully
09:00:55 INFO   [dora]: daemon dataflow finished on machine `01a162d7-5910-72fc-b5dd-3a860b05059d`
2026-10-22T09:00:55.457031Z  INFO run_inner: dora_daemon: exiting daemon because all required dataflows are finished health_check_interval_duration=None self.daemon_id=DaemonId { machine_id: None, uuid: 01a162d7-5910-72fc-b5dd-3a860b05059d }