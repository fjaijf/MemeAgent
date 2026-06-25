import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / "Agent" / ".env")
load_dotenv(PROJECT_ROOT / "Agent" / ".env.local", override=True)

api_key = (
    os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("QWEN_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
if not api_key:
    raise RuntimeError(
        "Set DASHSCOPE_API_KEY, QWEN_API_KEY, or OPENAI_API_KEY before running this test."
    )

client = OpenAI(
    # 各地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    api_key=api_key,
    # 以下为华北2（北京）地域的URL，各地域的URL不同。
    base_url=os.getenv(
        "OPENAI_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).strip().strip('"').strip("'"),
)

completion = client.chat.completions.create(
    model=os.getenv("MEMEAGENT_MODEL", "qwen3.7-plus").strip(),
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                    },
                },
                {"type": "text", "text": "图中描绘的是什么景象?"},
            ],
        },
    ],
)
print(completion.choices[0].message.content)
