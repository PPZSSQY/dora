from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent
W, H = 2400, 1550

BG = "#F4F6F8"
PANEL = "#FFFFFF"
INK = "#172033"
MUTED = "#4F5B6E"
BORDER = "#D9DEE7"
CUDA = "#2878C7"
CUDA_SOFT = "#EAF3FC"
NPU = "#D56A22"
NPU_SOFT = "#FFF0E5"
HOST = "#687386"

FONT = r"C:\Windows\Fonts\NotoSansSC-VF.ttf"


def font(size: int, bold: bool = False):
    # The variable font renders Chinese and Latin consistently. Pillow does
    # not expose weight axes portably, so headings use a larger size instead.
    return ImageFont.truetype(FONT, size=size)


F_TITLE = font(52)
F_SUBTITLE = font(27)
F_PANEL = font(38)
F_ROUTE = font(25)
F_STEP_TITLE = font(29)
F_BODY = font(25)
F_NUM = font(27)
F_FOOT = font(23)


def rounded(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, x, y1, y2, color):
    draw.line((x, y1, x, y2 - 10), fill=color, width=4)
    draw.polygon([(x - 9, y2 - 16), (x + 9, y2 - 16), (x, y2)], fill=color)


def wrap(draw, text, fnt, max_width):
    lines = []
    current = ""
    # Keep API identifiers and short English phrases intact while allowing
    # Chinese text to wrap naturally one glyph at a time.
    tokens = re.findall(r"[A-Za-z0-9_.'=:+→←⇄/()-]+|[\u3400-\u9fff]|\s+|.", text)
    for token in tokens:
        candidate = current + token
        if current and draw.textbbox((0, 0), candidate, font=fnt)[2] > max_width:
            lines.append(current)
            current = token.lstrip()
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def draw_panel(draw, x, y, w, h, spec, accent, soft):
    rounded(draw, (x, y, x + w, y + h), 28, PANEL, BORDER, 2)

    rounded(draw, (x + 34, y + 28, x + 260, y + 82), 27, accent)
    draw.text((x + 147, y + 55), spec["badge"], font=F_ROUTE, fill="white", anchor="mm")
    draw.text((x + 300, y + 48), spec["title"], font=F_PANEL, fill=INK, anchor="lm")
    draw.text((x + 38, y + 108), spec["route"], font=F_ROUTE, fill=MUTED)

    box_x = x + 38
    box_w = w - 76
    box_h = 178
    gap = 16
    first_y = y + 155

    for idx, step in enumerate(spec["steps"], 1):
        by = first_y + (idx - 1) * (box_h + gap)
        rounded(draw, (box_x, by, box_x + box_w, by + box_h), 18, soft, None)
        draw.rounded_rectangle(
            (box_x, by, box_x + 10, by + box_h), radius=5, fill=accent
        )
        cx, cy = box_x + 52, by + 46
        draw.ellipse((cx - 24, cy - 24, cx + 24, cy + 24), fill=accent)
        draw.text((cx, cy - 1), str(idx), font=F_NUM, fill="white", anchor="mm")
        draw.text((box_x + 92, by + 24), step[0], font=F_STEP_TITLE, fill=INK)
        body_lines = wrap(draw, step[1], F_BODY, box_w - 130)
        for li, line in enumerate(body_lines[:3]):
            draw.text(
                (box_x + 92, by + 75 + li * 36), line, font=F_BODY, fill=MUTED
            )
        if idx < len(spec["steps"]):
            arrow(draw, x + w // 2, by + box_h + 2, by + box_h + gap - 2, accent)

    fy = y + h - 58
    draw.line((x + 38, fy - 14, x + w - 38, fy - 14), fill=BORDER, width=2)
    draw.text((x + 38, fy), spec["footer"], font=F_FOOT, fill=HOST)


def render(filename, title, subtitle, left, right):
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)

    draw.text((W // 2, 62), title, font=F_TITLE, fill=INK, anchor="ma")
    draw.text((W // 2, 127), subtitle, font=F_SUBTITLE, fill=MUTED, anchor="ma")

    panel_y, panel_h, panel_w = 178, 1305, 1070
    draw_panel(draw, 80, panel_y, panel_w, panel_h, left, CUDA, CUDA_SOFT)
    draw_panel(draw, 1250, panel_y, panel_w, panel_h, right, NPU, NPU_SOFT)

    draw.line((1200, 220, 1200, 1452), fill=BORDER, width=2)
    image.save(OUT / filename, dpi=(180, 180), optimize=True)


def main():
    common_cpu_footer = "数据面：设备 → /dev/shm；接收端持有 CPU 零拷贝视图｜写入由 seqlock 发布"
    common_ipc_footer_cuda = "控制面：DORADMA 头 + 64 B IPC handle｜数据面：同一块 CUDA 设备池"
    common_ipc_footer_npu = "控制面：DORADMA 头 + 65 B IPC key｜数据面：同一块 Ascend 设备池"

    render(
        "01-cuda2cpu-vs-npu2cpu.png",
        "设备到主机：CUDA→CPU 与 NPU→CPU",
        "首帧注册一次；后续帧原位覆盖同一块主机共享内存",
        {
            "badge": "CUDA→CPU",
            "title": "cuda2cpu",
            "route": "发送进程（CUDA）  →  /dev/shm  →  接收进程（CPU）",
            "steps": [
                ("发送端造张量", "在 CUDA 上创建张量，get_tensor_info 取得设备指针、大小、形状与 dtype。"),
                ("注册并落首帧", "创建带完整数据区的 DORADMA shmem；cudaMemcpy D2H 把首帧写入数据区。"),
                ("发布主机描述", "头部记录 size / shape / dtype，device_type=CPU；此路径不导出 CUDA IPC。"),
                ("接收端零拷贝读取", "read_tensor_pool 返回 shmem 数据指针；tensor_from_info 包成 CPU tensor 视图。"),
                ("后续帧覆盖", "write_tensor_pool 再做 cudaMemcpy D2H，原位覆盖同一数据区并更新 seqlock。"),
            ],
            "footer": common_cpu_footer,
        },
        {
            "badge": "NPU→CPU",
            "title": "npu2cpu",
            "route": "发送进程（Ascend NPU）  →  /dev/shm  →  接收进程（CPU）",
            "steps": [
                ("发送端造张量", "在 NPU 上创建张量；torch_npu 初始化运行时并选择源卡。"),
                ("注册并落首帧", "创建完整 shmem；aclrtSynchronizeDevice 后 aclrtMemcpy D2H 写入首帧。"),
                ("发布主机描述", "头部记录 size / shape / dtype，device_type=CPU；此路径不导出 Ascend IPC key。"),
                ("接收端零拷贝读取", "read_tensor_pool 返回 shmem 指针；tensor_from_info 包成 CPU tensor 视图。"),
                ("后续帧覆盖", "write_tensor_pool 再做 aclrtMemcpy D2H，原位覆盖同一数据区并更新 seqlock。"),
            ],
            "footer": common_cpu_footer,
        },
    )

    render(
        "02-cpu2cuda-vs-cpu2npu.png",
        "主机到设备：CPU→CUDA 与 CPU→NPU",
        "设备池在发送进程中建立；接收进程只导入同一块设备内存",
        {
            "badge": "CPU→CUDA",
            "title": "cpu2cuda",
            "route": "发送进程（CPU）  →  CUDA 设备池  ⇄IPC⇄  接收进程（CUDA）",
            "steps": [
                ("发送端造张量", "在 CPU 上创建张量，get_tensor_info 暴露主机指针。"),
                ("注册设备池 + 首帧 H2D", "在接收卡 cudaMalloc 池；dma_copy 用 cudaMemcpy H2D 写入首帧，大张量可临时 pin。"),
                ("导出 64 B 句柄", "cudaIpcGetMemHandle 导出句柄，写入主机 shmem 头，device_type=CUDA。"),
                ("接收进程映射", "cudaIpcOpenMemHandle 导入同一设备池；torch.as_tensor 包成 CUDA tensor，不再复制。"),
                ("后续帧覆盖", "write_tensor_pool 重复 dma_copy H2D，覆盖同一 CUDA 池缓冲。"),
            ],
            "footer": common_ipc_footer_cuda,
        },
        {
            "badge": "CPU→NPU",
            "title": "cpu2npu",
            "route": "发送进程（CPU）  →  Ascend 设备池  ⇄IPC⇄  接收进程（NPU）",
            "steps": [
                ("发送端造张量", "在 CPU 上创建张量：torch.tensor(..., device='cpu')。"),
                ("注册设备池 + 首帧 H2D", "在接收用 NPU 上 aclrtMalloc；立刻 aclrtMemcpy H2D 把首帧写入池。"),
                ("导出 65 B 钥匙", "aclrtIpcMemGetExportKey 导出 key，写入主机 shmem 头，device_type=Ascend。"),
                ("接收进程映射", "aclrtIpcMemImportByKey 映射同一 NPU 池；from_dlpack 包成 NPU tensor，不再复制。"),
                ("后续帧覆盖", "write_tensor_pool 再做 aclrtMemcpy H2D，覆盖同一 NPU 池缓冲。"),
            ],
            "footer": common_ipc_footer_npu,
        },
    )

    render(
        "03-cuda-inner-vs-npu-inner.png",
        "同卡设备内：CUDA inner 与 NPU inner",
        "源张量与池缓冲位于同一张卡；接收进程通过 IPC 共享池缓冲",
        {
            "badge": "CUDA INNER",
            "title": "cuda inner",
            "route": "发送 CUDA:0  →  同卡池缓冲  ⇄IPC⇄  接收 CUDA:0",
            "steps": [
                ("同卡源张量", "发送端在 cuda:0 上创建源 tensor。"),
                ("建池 + 首帧 D2D", "cudaMalloc 创建池缓冲；cudaMemcpy D2D 把首帧复制进池。"),
                ("导出 64 B 句柄", "cudaIpcGetMemHandle 写入 DORADMA 头，设备数据不落入 shmem。"),
                ("接收进程零拷贝映射", "cudaIpcOpenMemHandle 导入池指针；__cuda_array_interface__ 包成 CUDA tensor。"),
                ("后续帧同卡覆盖", "_cuda_memcpy_gpu_buf 使用 D2D 覆盖同一池缓冲，接收视图保持不变。"),
            ],
            "footer": common_ipc_footer_cuda,
        },
        {
            "badge": "NPU INNER",
            "title": "npu inner",
            "route": "发送 NPU:0  →  同卡池缓冲  ⇄IPC⇄  接收 NPU:0",
            "steps": [
                ("同卡源张量", "发送端在 npu:0 上创建源 tensor。"),
                ("建池 + 首帧 D2D", "aclrtMalloc 创建池；把两端指针包成 uint8 tensor，用 torch.copy_ 快速写首帧。"),
                ("导出 65 B 钥匙", "aclrtIpcMemGetExportKey 写入 DORADMA 头，设备数据不落入 shmem。"),
                ("接收进程零拷贝映射", "aclrtIpcMemImportByKey 导入池指针；from_dlpack 包成 NPU tensor。"),
                ("后续帧同卡覆盖", "_npu_same_device_d2d 再用 torch.copy_ 覆盖同一池缓冲。"),
            ],
            "footer": common_ipc_footer_npu,
        },
    )

    render(
        "04-cuda2cuda-vs-npu2npu.png",
        "跨卡设备间：CUDA→CUDA 与 NPU→NPU",
        "优先设备 P2P；不可用时回退到页锁主机中转，并在后续帧复用所选路径",
        {
            "badge": "CUDA→CUDA",
            "title": "cuda2cuda",
            "route": "发送 CUDA:0  →  P2P / host transit  →  接收 CUDA:1",
            "steps": [
                ("跨卡源张量", "发送端在 cuda:0 上创建源 tensor，目标为 cuda:1。"),
                ("选择首帧路径", "有 P2P：启用 peer access 并 D2D；无 P2P：页锁 host 缓冲 D2H→H2D。"),
                ("导出 64 B 句柄", "cudaIpcGetMemHandle 导出池句柄并写入 DORADMA 头。"),
                ("接收卡导入映射", "接收进程 cudaIpcOpenMemHandle 导入；torch.as_tensor 形成 cuda:1 视图。"),
                ("后续帧复用路径", "write_tensor_pool 沿注册时路径覆盖同一池：D2D 或 host transit。"),
            ],
            "footer": common_ipc_footer_cuda,
        },
        {
            "badge": "NPU→NPU",
            "title": "npu2npu",
            "route": "发送 NPU:0  →  P2P / host transit  →  接收 NPU:1",
            "steps": [
                ("跨卡源张量", "发送端在 npu:0 上创建源 tensor，目标为 npu:1。"),
                ("选择首帧路径", "接收卡 aclrtMalloc；有 P2P 用 aclrtMemcpy D2D，否则 aclrtMallocHost 做 D2H→H2D。"),
                ("导出 65 B 钥匙", "aclrtIpcMemGetExportKey 导出 key 并写入 DORADMA 头。"),
                ("接收卡导入映射", "aclrtIpcMemImportByKey 导入同一池；from_dlpack 形成 npu:1 tensor。"),
                ("后续帧复用路径", "write_tensor_pool 沿注册时路径覆盖同一池：P2P D2D 或 host transit。"),
            ],
            "footer": common_ipc_footer_npu,
        },
    )


if __name__ == "__main__":
    main()
