import pandas as pd
import numpy as np

# 创建 Series
s = pd.Series([1, 2, 3, 4, 5])
print(s)

# 使用 lambda 函数
result = s.apply(lambda x: x * 2)
print(result)