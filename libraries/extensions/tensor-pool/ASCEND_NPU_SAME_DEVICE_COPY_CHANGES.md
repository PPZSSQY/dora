# 昇腾同卡 D2D 快拷改动记录

## 范围

只改单机同卡 `npu:0 → npu:0` 往池里填数的方式。

不改：

- 开池（仍 `aclrtMalloc`）和钥匙（仍 `GetExportKey` / `ImportByKey`）
- 跨卡 P2P D2D、host transit
- CPU→NPU H2D、NPU→CPU、CUDA
- README、环境安装

## 原因

`aclrtMemcpy(..., DEVICE_TO_DEVICE)` 同卡实测约 3700 MB/s；`torch.copy_` 同卡约 47 万 MB/s。`npu_inner` 原先走慢的那条。

## 改动文件

### `python/src/transport.rs`（Ascend helper）

新增 `_npu_same_device_d2d`：把源指针和池指针用已有的 `_wrap_npu_ptr`（`from_dlpack`）包成 `uint8` 一维张量，`dst.copy_(src)`，再 `torch.npu.synchronize()`。

#### `_setup_npu_ipc_from_npu`

原逻辑：`path == "same-device D2D"` 时 `aclrtMemcpy` D2D。

现逻辑：同路径改走 `_npu_same_device_d2d`。`P2P D2D` / `host transit` 仍 `aclrtMemcpy`。

#### `_npu_memcpy_npu_to_pool`

原逻辑：无中转时一律 `aclrtMemcpy` D2D（同卡和跨卡 P2P 共用）。

现逻辑：

- 有 `transit_ptr`：仍 D2H + H2D
- `src_dev != dst_dev`：仍 P2P `aclrtMemcpy` D2D
- 同卡：`_npu_same_device_d2d`

### `examples/npu_inner.yml`

沿用，作为同卡实机样例（不新建 dataflow）。

### `examples/bench_h2d_d2d.py`

增加一项：`torch` 源 + `aclrtMalloc` 目的，`copy_`。用来确认「混合内存」是否仍走快通道。

## 验证

本地不跑 NPU。实机：

```bash
python examples/bench_h2d_d2d.py --device npu:0
dora run libraries/extensions/tensor-pool/examples/npu_inner.yml
```

期望：`torch.copy_ into aclrtMalloc` 接近 `torch D2D`；`npu_inner` 仍打印 `same-device D2D`，吞吐明显高于原先约 3000 MB/s，预览张量一致。跨卡 `npu2npu.yml` / `npu2npu_nop2p.yml` 行为不变。
