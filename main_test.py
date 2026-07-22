import json
import os
from pathlib import Path

import lark_oapi as lark
from dotenv import load_dotenv
from lark_oapi.api.im.v1 import (
    P2ImMessageReceiveV1,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)
from lark_oapi.event.dispatcher_handler import EventDispatcherHandler

# 加载 .env 文件（与当前模块同目录）
load_dotenv(Path(__file__).resolve().parent / ".env")

# App 凭据 —— 从环境变量读取
APP_ID = os.getenv("FEISHU_APP_ID")
APP_SECRET = os.getenv("FEISHU_APP_SECRET")


if not APP_ID or not APP_SECRET:
    raise RuntimeError(
        "缺少飞书应用凭据。请在 app/.env 中配置 FEISHU_APP_ID 和 FEISHU_APP_SECRET，"
        "或设置对应的环境变量。"
    )


# 1. 定义消息处理函数
def do_message_receive(data: P2ImMessageReceiveV1) -> None:
    """处理接收到的消息事件，并回复 echo 消息。"""
    # 1. 从事件中提取消息对象（运行时事件必定包含 event.message）
    event_data = data.event
    assert event_data is not None, "Event data is missing"
    message = event_data.message
    assert message is not None, "Message is missing"
    assert message.chat_id is not None, "chat_id is missing"
    assert message.content is not None, "content is missing"
    assert message.message_id is not None, "message_id is missing"

    # 2. 提取 chat_id 和原始消息内容（Feishu 消息内容是 JSON 字符串）
    chat_id = message.chat_id
    content_json = json.loads(message.content)
    text = content_json.get("text", "")

    # 3. 构造回复内容：回显收到的消息
    reply_text = f"收到你的消息：「{text}」"
    reply_content = json.dumps({"text": reply_text})

    # 4. 通过 Lark REST API 发送回复消息
    body = ReplyMessageRequestBody.builder() \
        .content(reply_content) \
        .msg_type("text") \
        .build()

    request = ReplyMessageRequest.builder() \
        .message_id(message.message_id) \
        .request_body(body) \
        .build()

    client = lark.Client.builder() \
        .app_id(APP_ID) \
        .app_secret(APP_SECRET) \
        .build()

    assert client.im is not None, "IM service is not available"
    response = client.im.v1.message.reply(request)
    if not response.success():
        print(f"Failed to send reply: log_id={response.get_log_id()}")

    # 5. 打印完整事件数据供调试
    print(lark.JSON.marshal(data))


# 3. 注册事件处理器
event_handler = (
    EventDispatcherHandler.builder(APP_ID, APP_SECRET)
    .register_p2_im_message_receive_v1(do_message_receive)
    .build()
)


def main() -> None:
    # 4. 创建长连接客户端并启动
    cli = lark.ws.Client(
        APP_ID,
        APP_SECRET,
        event_handler=event_handler,
        log_level=lark.LogLevel.DEBUG,
    )
    cli.start()


if __name__ == "__main__":
    main()
