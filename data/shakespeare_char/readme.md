
# 小莎士比亚，字符级

小莎士比亚，著名的char-rnn的好老朋友 :) 在字符级上处理。

在运行完 `prepare.py`后，你会得到以下文件:

- train.bin has 1,003,854 tokens
- val.bin has 111,540 tokens
> token是文本数据的基本单位，在字符级模型中，每个token通常对应一个字符。这些文件包含了文本数据的编码版本，供模型训练使用。
> **注意**: 这些文件是二进制格式的，不能直接打开。它们包含了文本数据的编码版本，供模型训练使用。
> 你可以使用以下代码来加载这些文件并查看其中的内容：

```python
import numpy as np

# Load the binary files
train_data = np.fromfile('train.bin', dtype=np.uint16)
val_data = np.fromfile('val.bin', dtype=np.uint16)

# Print some statistics
print(f"Length of train data: {len(train_data)} tokens")
print(f"Length of val data: {len(val_data)} tokens")

# Print first few tokens
print("First 100 tokens of train data:")
print(train_data[:100])
```