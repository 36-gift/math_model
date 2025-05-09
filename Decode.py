import chardet

# 替换为你的文件路径
#file_path = r"D:\2025美赛论文\数据\Complete_Historical_Stone_Step_Data (1).csv"

#file_path = r"D:\python_application\web_spider\000001_SZSE.csv"
#file_path = r"D:\python_application\web_spider\300750_SZSE.csv"
#file_path = r"D:\python_application\web_spider\600519_SHE_贵州茅台.csv"
#file_path = r"D:\python_application\web_spider\600760_SHE_中航沈飞.csv"
file_path = r"D:\python_application\web_spider\600036_SHE_招商银行.csv"

# 读取文件内容
with open(file_path, 'rb') as f:
    raw_data = f.read()

# 检测编码
result = chardet.detect(raw_data)
print(f"Detected encoding: {result['encoding']} with confidence: {result['confidence']}")


# #检测文件路径是否存在
# import os
#
# file_path = r"D:\python_application\web_spider\000001_SZSE.csv"
#
# if os.path.exists(file_path):
#     print("文件路径正确，文件存在。")
# else:
#     print("文件路径错误或文件不存在。")