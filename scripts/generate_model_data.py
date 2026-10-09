"""将全 INT8 TFLite 模型转换为可直接参与 C++ 编译的字节数组。"""

from pathlib import Path

# 始终相对于本脚本定位仓库根目录，而不是依赖命令执行时的当前目录。这样既可以在
# 仓库根目录运行 ``python scripts/generate_model_data.py``，也可以从其他目录调用脚本。
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPOSITORY_ROOT / "models/exported/best_model_int8_baseline.tflite"
HEADER_PATH = REPOSITORY_ROOT / "firmware/model_data.h"
SOURCE_PATH = REPOSITORY_ROOT / "firmware/model_data.cc"

# 每行放置固定数量的字节，避免生成超长的 C++ 源代码行，同时保持 Git diff 可读。
BYTES_PER_LINE = 12

HEADER_TEMPLATE = """\
#pragma once
#include <cstddef>
#include <cstdint>

// TFLite FlatBuffer 的只读字节数组及其长度。数组定义在 model_data.cc 中，避免每个
// 包含本头文件的编译单元都复制一份模型数据。
extern const std::uint8_t g_model_data[];
extern const std::size_t g_model_data_len;
"""


def _format_model_bytes(data: bytes) -> str:
    """把模型二进制格式化为多行十六进制 C++ 初始化列表。

    ``data`` 中的每个元素都是一个取值范围为 0～255 的字节。这里将每个字节转换为
    ``0x00``～``0xff`` 形式的 C++ 整数字面量，并按 ``BYTES_PER_LINE`` 个字节分行。
    返回值只包含数组初始化列表内部的文本，不包含变量声明和外层花括号。

    例如，输入 ``b"\x20\x00\xff"`` 将生成：

    ``    0x20, 0x00, 0xff,``

    每个值末尾都保留逗号；C++ 允许初始化列表的最后一个元素带尾随逗号，因此最后一行
    不需要单独处理，后续重新生成模型时也能得到更整洁、稳定的代码差异。
    """
    # 先保存每一行的文本，最后一次性用 LF 连接。这样不会在循环中反复拼接越来越长的
    # 字符串，也能由调用方统一控制整个生成文件末尾的换行。
    lines = []

    # offset 依次取 0、BYTES_PER_LINE、2 * BYTES_PER_LINE……，代表当前行在原始
    # FlatBuffer 中的起始字节位置。
    for offset in range(0, len(data), BYTES_PER_LINE):
        # bytes 切片在最后不足 BYTES_PER_LINE 个字节时会自动返回剩余内容，因此模型
        # 长度不必是每行字节数的整数倍，也不会补入任何不属于模型的填充字节。
        chunk = data[offset : offset + BYTES_PER_LINE]

        # ``02x`` 表示使用小写十六进制并至少补齐两位。例如十进制 5 会变为 ``05``，
        # 加上 ``0x`` 前缀后得到合法且宽度一致的 C++ 字节字面量 ``0x05``。
        values = ", ".join(f"0x{value:02x}" for value in chunk)

        # 四个空格仅用于提高生成的 C++ 源码可读性，不会进入最终编译出的模型数据。
        lines.append(f"    {values},")

    # 使用显式 ``\n`` 保证生成内容与操作系统无关；文件写入阶段同样固定使用 LF。
    return "\n".join(lines)


def generate_model_data(
    model_path: Path = MODEL_PATH,
    header_path: Path = HEADER_PATH,
    source_path: Path = SOURCE_PATH,
) -> int:
    """读取 TFLite FlatBuffer，并生成对应的 C++ 头文件和源文件。

    返回原始模型的字节数，供命令行入口打印生成摘要。
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"TFLite model does not exist: {model_path}")

    data = model_path.read_bytes()
    if not data:
        raise ValueError(f"TFLite model is empty: {model_path}")

    # alignas(16) 为模型首地址提供稳定对齐，满足桌面 TFLM 访问 FlatBuffer 时的对齐
    # 要求，也便于后续迁移到对齐要求更严格的嵌入式平台。
    source = f"""\
#include "model_data.h"

alignas(16) const std::uint8_t g_model_data[] = {{
{_format_model_bytes(data)}
}};

const std::size_t g_model_data_len = sizeof(g_model_data);
"""

    # firmware 目录通常已经存在；显式创建父目录使脚本在干净检出或自定义输出路径下
    # 也能可靠运行。固定使用 LF 换行，避免生成文件因主机平台不同产生无意义差异。
    header_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(HEADER_TEMPLATE, encoding="utf-8", newline="\n")
    source_path.write_text(source, encoding="utf-8", newline="\n")

    return len(data)


def main() -> None:
    """生成默认固件模型文件并打印可核对的结果摘要。"""
    model_size = generate_model_data()
    print(f"Generated: {HEADER_PATH}")
    print(f"Generated: {SOURCE_PATH}")
    print(f"Model size: {model_size} bytes")


if __name__ == "__main__":
    main()
