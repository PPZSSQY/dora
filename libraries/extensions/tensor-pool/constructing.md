cd /workspace/tensorpool-ascend/dora
cd libraries/extensions/tensor-pool
source .venv/bin/activate



deactivate

# Python 绑定（含 NPU→CPU 传输）
cd /workspace/tensorpool-ascend/dora
maturin develop -m apis/python/node/Cargo.toml --features tensor-pool

# CLI（dora run 用）

cargo build -p dora-cli --features tensor-pool
export PATH=/workspace/tensorpool-ascend/dora/target/debug:$PATH

dora run /workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool/examples/npu2npu_nop2p.yml
















# tensor-pool 结构说明（昇腾迁移备忘）

> 先不写代码，只讲清楚：各文件干什么、整体干什么、和 dora 什么关系、迁移要改哪里。

---

## 一句话：整体在干什么？

**大张量别走普通消息总线，放进一块共享内存（池），节点之间只传一个「池 ID」。**

相机一帧 8MB，若每帧都当普通消息拷一遍，又慢又费。tensor-pool 的做法是：

1. 发送方注册一块共享内存池，把 tensor 写进去；
2. 只把池的 ID（很小）通过 dora 正常输出发出去；
3. 接收方拿 ID 去映射同一块内存（或跨机镜像），直接读数据。

同机多为 `/dev/shm` 零拷贝；若源/目标在 NVIDIA GPU 上，还会走 CUDA pinned / DMA / IPC。跨机则由 daemon 侧做镜像 + TCP/zenoh 传数据。

**它是可选扩展，不在 dora 1.0 稳定 API 里，随时可能改。**

---

## 和主体 dora 是什么关系？说人话

| | dora 主体 | tensor-pool |
|---|---|---|
| 知道什么 | 节点、数据流、定时器、普通消息、扩展通道 | 池、共享内存布局、seqlock、CUDA、跨机镜像 |
| 怎么连 | 提供通用扩展口：`extension_store/load/drop`、`extension_request`、守护进程间 `ExtensionMessage` | 把「池描述符 / 请求」塞进这些口里，dora **当黑盒字节** 转发，不解析 |
| 默认构建 | 不开 `tensor-pool` feature → 根本不编译这块 | 要显式 `--features tensor-pool` 才进 Python Node / daemon / CLI |
| 比喻 | 快递公司：只认「包裹 + 单号」 | 你自己设计的「大货仓库 + 仓位号」；快递只帮你递仓位号和仓库间的控制信 |

**结论：尽量只改本目录；dora 核心只做「接线」（feature 门控 + `DaemonServices`），不要把 Ascend/NPU 细节塞进 dora 主体。**

接线位置（一般不用为昇腾大改，知道即可）：

- `apis/python/node`：`Node.register/write/read/free_tensor_pool`（feature 打开才有）
- `binaries/daemon/src/pool_extension.rs`：daemon 把扩展挂上
- `binaries/cli`：把 feature 传给内嵌 daemon

---

## 目录地图：各块代码在干什么

```
tensor-pool/
├── src/                          # Rust：池表 + daemon 半边
├── python/                       # 节点侧传输 + Python 辅助
├── examples/                     # 可跑的数据流样例
├── Cargo.toml / README.md        # 特性开关与说明
└── requirements.txt / .venv      # 本机已装的运行环境（迁移无关逻辑）
```

### 1. `src/` — Rust 库（池登记册 + 跨机 daemon 半边）

| 路径 | 干什么 |
|---|---|
| `src/lib.rs` | **池总表**：谁注册了哪块池、谁碰过、崩溃后怎么回收 `/dev/shm`；不管 CUDA |
| `src/protocol.rs` | **本扩展自己的报文**：节点↔本机 daemon、daemon↔对端 daemon；用 postcard 塞进 dora 的不透明通道；命名空间 `dora-tensor-pool` |
| `src/daemon/mod.rs` | daemon 半边入口；环境变量 `DORA_MEMORY_POOL_CROSS_MACHINE` 总开关 |
| `src/daemon/services.rs` | **向 dora 要的能力清单**（trait）：machine_id、zenoh、扩展表存删等；扩展不能直接摸 daemon 内部状态 |
| `src/daemon/handlers.rs` | 收到不透明字节后解码，处理注册/写/释放、对端消息 |
| `src/daemon/state.rs` | 本扩展在 daemon 里挂着的状态（订阅、跨机池记录等） |
| `src/daemon/mirror.rs` | **镜像段**：在本机再建一份共享内存、写 DORADMA 头、seqlock；跨机接收方读本地镜像 |
| `src/daemon/data_plane.rs` | **跨机大数据通道**：direct-TCP，把 tensor 字节打进镜像段 |
| `src/daemon/control.rs` | **控制面**：zenoh 上发 Register/Write/Ack/Free 等控制消息 |
| `src/daemon/auth.rs` | direct-TCP 可选共享口令握手（`DORA_MEMORY_POOL_AUTH_TOKEN`） |

人话：**`lib.rs` 管「本机池名册」；`daemon/*` 管「跨进程/跨机怎么把大块数据搬过去」。CPU 共享内存路径基本在这里，不绑 NVIDIA。**

### 2. `python/` — 节点里真正「建池 / 读写」的地方（昇腾重点）

| 路径 | 干什么 |
|---|---|
| `python/src/lib.rs` | 模块入口，导出 `Pool` |
| `python/src/seam.rs` | **和 dora 扩展通道的接头**：把池描述符 JSON 编解码，调用 `extension_store/load/drop` |
| `python/src/transport.rs` | **核心运输逻辑**（最大、最硬）：共享内存段布局、seqlock、注册/写/读/释放；内嵌一段 ctypes 调 `libcudart.so` 做 pin / DMA / IPC / `cudaMemcpy` |
| `python/dora_tensor_pool/__init__.py` | 对外导出辅助函数 |
| `python/dora_tensor_pool/tensor_info.py` | `get_tensor_info` / `tensor_from_info`：torch tensor ↔ 池要的元信息（指针、dtype、shape、device） |
| `python/dora_tensor_pool/cuda.py` | **纯 CUDA 工具**：`libcudart` IPC、host register 等；给 GPU↔GPU / 打开对端显存句柄用 |
| `python/pyproject.toml` / `Cargo.toml` | Python 包装与 Rust crate 依赖声明 |

人话：**业务节点调的四个方法，最终都落到 `transport.rs`；凡是写着 cuda / libcudart / IPC 的，都是 NVIDIA 路径。**

### 3. `examples/` — 怎么用、测什么

| 路径 | 干什么 |
|---|---|
| `sender.py` / `receiver.py` | 发/收：注册池、写帧、读帧、测吞吐；`sender_device` / `receiver_device` 可读 `cpu` 或 `cuda:0` |
| `cpu2cpu.yml` 及 negative 几个 yml | **纯 CPU**，无 GPU 也能跑（迁移时优先保这条） |
| `cpu2cuda.yml` / `cuda2cpu.yml` / `cuda2cuda*.yml` | 涉及 CUDA 设备的吞吐路径 |
| `*_cross*.yml` | 跨 daemon / 跨机 |
| `examples/README.md` | 怎么装依赖、怎么跑、期望吞吐与告警 |

---

## 数据怎么走（心智模型）

```
[sender 节点]
  torch tensor
    → get_tensor_info
    → register_tensor_pool / write_tensor_pool   (transport.rs)
    → 写 /dev/shm 段 +（可选）CUDA pin/IPC
    → extension_store 存描述符；send_output 只发 pool_id

[dora]
  普通消息传 pool_id；扩展表存/取不透明描述符；跨机时 extension_request / ExtensionMessage

[receiver 节点]
  收到 pool_id
    → read_tensor_pool → 映射 shmem / 开 IPC
    → tensor_from_info → torch tensor

[跨机时再多一步]
  本机 daemon 读发送方段 → TCP/zenoh → 对端 daemon 写入镜像段 → 接收节点读本地镜像
```

---

## 昇腾迁移：你大概要改哪些代码？

### 原则

1. **先保 CPU↔CPU**（`cpu2cpu.yml`）：确认共享内存 + 四个 API + daemon 回收在昇腾机器上可用。  
2. **再做 NPU 路径**：用 CANN/ACL（或 `torch_npu`）替换 CUDA 语义，不要幻想改两行 device 字符串就完事。  
3. **尽量不改 dora 主体接线**；改本扩展内部实现。

### 必改 / 高概率必改（NPU 零拷贝或设备侧传输）

| 优先级 | 文件 | 为什么 |
|---|---|---|
| ★★★ | `python/src/transport.rs` | 内嵌 `libcudart`、设备名 `cuda*`、pin/DMA/IPC/DtoD 分支全在这里；昇腾要对齐 ACL 内存注册、H2D/D2H、进程间共享（若有） |
| ★★★ | `python/dora_tensor_pool/cuda.py` | 整文件是 CUDA runtime；需昇腾等价实现（可新建 `npu.py` / `ascend.py`，再改导出） |
| ★★ | `python/dora_tensor_pool/tensor_info.py` | device 字符串、从 NPU tensor 取 ptr/`__cuda_array_interface__` 类接口（昇腾是另一套 array interface） |
| ★★ | `python/dora_tensor_pool/__init__.py` | 导出面随上面模块调整 |
| ★ | `examples/sender.py` / `receiver.py` | `cuda:0` → `npu:0`（或你们约定的 device 名）；断言 `torch.cuda` → `torch.npu` / ACL |
| ★ | `examples/*cuda*.yml` | 设备环境变量、依赖安装（torch_npu 等）；可先复制为 `*npu*.yml` |

### 通常不用为「昇腾」大改（CPU/跨机控制面）

| 文件 | 说明 |
|---|---|
| `src/lib.rs`、`src/protocol.rs` | 池表与报文；与 GPU 厂商无关 |
| `src/daemon/*` | 跨机镜像/TCP/zenoh；搬的是字节，不绑 CUDA（除非你以后要做「NPU 显存直通跨机」，那是更大课题） |
| `python/src/seam.rs` | 只编解码描述符；若描述符里增加 `npu` 字段，可能小改字段，逻辑框架可留 |
| dora 的 `pool_extension.rs` 等接线 | 继续 feature 挂载即可 |

### 描述符 / 元数据里和 CUDA 绑死的概念（改 transport 时一起理清）

- `pinned_type`: 现为 `"cpu"` / `"cuda"` → 可能要加 `"npu"`  
- `ipc_present` + 段头里的 CUDA IPC handle → 昇腾若无对等 IPC，CPU 中转或厂商 IPC 要重新设计  
- `device` 以 `cuda` 前缀分支的逻辑 → 改为识别 `npu` / `ascend`

### 建议落地顺序（任务拆解）

1. **验证**：昇腾机上跑通 `cpu2cpu.yml`（及 negative 生命周期几个 yml）。  
2. **盘点**：列出 ACL/`torch_npu` 能否提供：host pin、H2D/D2H、同机跨进程共享设备内存。  
3. **替换**：`cuda.py` + `transport.rs` 里 CUDA 调用 → Ascend 实现；CPU 分支保持不动。  
4. **样例**：加 `cpu2npu` / `npu2cpu` / `npu2npu`，先单机再跨机。  
5. **文档/构建**：README、feature 说明、依赖（`torch_npu`、CANN）写入你们自己的安装说明。

### 明确「做不到就降级」的边界

昇腾若短期内没有稳定的「跨进程 GPU 显存 IPC」：

- **同机 NPU**：可先走「NPU → 拷到 pinned/普通 host → shmem → 对端再 H2D」，正确性优先，吞吐次之；  
- **跨机**：继续用现有 daemon CPU 镜像路径（本来 GPU 跨机也是 DtoH → 网络 → HtoD）。

---

## 本机环境（备忘，与逻辑无关）

```bash
cd /workspace/tensorpool-ascend/dora/libraries/extensions/tensor-pool
source .venv/bin/activate
export PATH=/workspace/tensorpool-ascend/dora/target/debug:$PATH
```

- `.venv`：Python 依赖（含 CPU 版 torch）  
- `PATH`：让 shell 找到带 `tensor-pool` feature 编出来的 `dora`  
- 本机无 NVIDIA CUDA；`torch.cuda.is_available()` 为假属预期  

---

## 一句话收束

**tensor-pool = 大张量共享内存运输插件；dora = 只帮传 ID 和不透明控制字节的快递。**  
迁昇腾时，**主战场是 `python/src/transport.rs` + `python/dora_tensor_pool/cuda.py`（及 tensor_info/样例）**；daemon 与协议层可先不动，用 CPU 路径验收底座。
