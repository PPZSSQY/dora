# Ascend NPU-to-CPU migration record

## Scope

This change implements only the single-machine `NPU -> CPU` tensor-pool path
for CANN 8.x/9.x and tensors created by `torch_npu`.

## Previous behavior

- Only device names beginning with `cuda` were treated as accelerator tensors.
- A `torch_npu` device such as `npu:0` fell through to the CPU branch.
- The CPU branch attempted `std::ptr::copy_nonoverlapping` with the NPU device
  pointer, which is invalid and could crash the process.
- All device-to-host copies were hard-wired to `libcudart.so` and
  `cudaMemcpy(..., cudaMemcpyDeviceToHost)`.

## Current behavior

- `npu` and `npu:<index>` are recognized explicitly.
- Ascend helpers are cached separately from CUDA helpers and load
  `libascendcl.so` only when an NPU source is used.
- Before reading an NPU tensor, `aclrtSynchronizeDevice` waits for pending
  `torch_npu` work. The payload is then copied synchronously into the existing
  `/dev/shm` data region with:

  ```text
  aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST)
  ```

- The registration copy and both fast/slow `write_tensor_pool` paths use this
  Ascend D2H implementation.
- CUDA behavior remains on the existing CUDA helper and is unchanged.
- NPU receivers and non-CPU destinations for an NPU source return a clear
  unsupported-path error instead of silently using the CPU branch.

## Runtime initialization

The example sender imports `torch_npu` and calls `torch.npu.set_device` before
creating the tensor. That initializes AscendCL for PyTorch.

The transport also initializes explicitly on the first NPU copy:

1. `aclInit(NULL)` once per process.
2. If `torch_npu` already called it, AscendCL returns
   `ACL_ERROR_REPEAT_INITIALIZE` (100002). The official `aclInit` contract says
   this repeat error is ignored and execution continues.
3. `aclrtSetDevice(index)` runs when the tensor's NPU index changes.
4. `aclFinalize` is not called. Finalizing here would destroy the runtime that
   `torch_npu` is still using; the process exit releases it.

## Deliberate limitations

- CPU-to-NPU is not implemented.
- NPU-to-NPU and Ascend IPC are not implemented.
- Cross-machine Ascend transport is not implemented.
- The destination `/dev/shm` mapping is pageable host memory. This first
  implementation favors correctness and compatibility; pinned-host
  optimization can be evaluated separately.

## Manual validation

Use `examples/npu2cpu.yml` on a machine with `torch_npu`, CANN 8.x/9.x, and
`libascendcl.so` available through the configured Ascend runtime environment.
No Ascend environment or dependency is installed by this change.
