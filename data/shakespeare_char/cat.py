import numpy as np
import os
import pickle

# 加载二进制文件
file_dir = os.path.dirname(__file__)
train_data = np.fromfile(os.path.join(file_dir, 'train.bin'), dtype=np.uint16)
val_data = np.fromfile(os.path.join(file_dir, 'val.bin'), dtype=np.uint16)

# 加载meta信息，获取itos映射
# meta.pkl 是一个字典，包含了itos（索引到字符的映射）和stoi（字符到索引的映射）
# itos 是一个列表，每个元素是一个字符
# stoi 是一个字典，键是字符，值是索引
with open(os.path.join(file_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
itos = meta['itos']

# 定义decode函数
def decode(l):
    return ''.join([itos[i] for i in l]) # 把索引转换为字符

# 打印token数量
print(f"Length of train data: {len(train_data)} tokens")
print(f"Length of val data: {len(val_data)} tokens")

# 打印前100个token
print("First 100 tokens of train data:")
print(train_data[:100])

# 打印转换后的字符
print("\nFirst 100 characters of train data:")
print(decode(train_data[:100]))