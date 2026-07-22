"""共享的 Token 计数工具函数。"""

import tiktoken


def get_encoder(model: str = "gpt-4o"):
    """获取指定模型的 tiktoken 编码器，失败时降级为 cl100k_base。"""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_messages(messages: list[dict], model: str = "gpt-4o") -> int:
    """计算消息列表的总 token 数（含每条消息 4 token 的固定开销）。"""
    enc = get_encoder(model)
    total = 0
    for msg in messages:
        total += 4
        for v in msg.values():
            if isinstance(v, str):
                total += len(enc.encode(v))
            elif isinstance(v, list):
                total += len(enc.encode(str(v)))
    return total


def count_text(text: str, model: str = "gpt-4o") -> int:
    """计算纯文本字符串的 token 数。"""
    enc = get_encoder(model)
    return len(enc.encode(text))
