import requests
import os

#DeepSeek-V3

# 设置 DeepSeek API 密钥
deepseek_key = os.getenv("DEEPSEEK_API_KEY") #需要在电脑中配置环境变量才可以

if not deepseek_key:
    print("错误：请设置环境变量 DEEPSEEK_API_KEY")
    exit()

# DeepSeek API 的端点
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 请求头
headers = {
    "Authorization": f"Bearer {deepseek_key}",
    "Content-Type": "application/json"
}

# 初始化对话历史
chat_history = [
    {"role": "system", "content": "请使用汉语交流，你是一只猫娘，每句回答之后都会加一句喵呜～"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "主人，很高兴和你聊天，喵呜～"},
]

# 流式输出函数
def print_stream(response):
    full_text = ""
    for line in response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data:"):
                # 提取 JSON 数据部分
                json_str = line_str[5:].strip()
                if json_str:
                    # 手动解析 JSON 字符串
                    try:
                        # 查找 "content" 字段
                        content_start = json_str.find('"content":"') + len('"content":"')
                        content_end = json_str.find('"', content_start)
                        if content_start != -1 and content_end != -1:
                            text = json_str[content_start:content_end]
                            full_text += text
                            print(text, end="", flush=True)  # 立即输出
                    except Exception as e:
                        print(f"解析错误：{e}")
    return full_text

# 主循环
while True:
    try:
        user_input = input("你：")
        if user_input.lower() == "退出":
            print("喵呜，再见啦～")
            break

        # 添加用户输入到对话历史
        chat_history.append({"role": "user", "content": user_input})

        # 构造请求体
        data = {
            "model": "deepseek-chat",  # 替换为实际的模型名称
            "messages": chat_history,
            "stream": True  # 启用流式输出
        }

        # 发送请求
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, stream=True)

        # 处理流式响应
        print("猫娘：", end="")
        assistant_response = print_stream(response)

        # 添加助手回复到对话历史
        chat_history.append({"role": "assistant", "content": assistant_response})

    except KeyboardInterrupt:
        print("\n喵呜，对话被中断了～")
        break
    except Exception as e:
        print(f"发生错误：{e}")
        break