import google.generativeai as genai

# 替换为您的 API 密钥
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KET")

genai.configure(api_key=GOOGLE_API_KEY)

def generate_text(prompt, model_name="Gemini 2.0 Flash Experimental", max_output_tokens=150, temperature=0.7):
  """
    使用 Gemini API 生成文本。

    Args:
        prompt (str): 输入提示文本。
        model_name (str, optional): 使用的模型名称。默认为 "gemini-pro"。
        max_output_tokens (int, optional): 生成文本的最大 token 数量。默认为 150。
        temperature (float, optional): 控制生成文本的随机性。默认为 0.7。

    Returns:
        str: 生成的文本。
  """
  try:
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
      prompt,
      generation_config=genai.types.GenerationConfig(
        max_output_tokens=max_output_tokens,
        temperature=temperature
      )
    )
    return response.text.strip()
  except Exception as e:
    print(f"An error occurred: {e}")
    return None


if __name__ == "__main__":
    # 您可以修改这里的 prompt 来测试不同的输入
    prompt = "请介绍一下机器学习的基本概念。"
    generated_text = generate_text(prompt)

    if generated_text:
        print("生成的文本:\n")
        print(generated_text)

