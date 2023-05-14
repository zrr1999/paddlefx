from __future__ import annotations

import pandas as pd

# 读取CSV文件为数据帧
df = pd.read_csv('classified_bytecode.txt')
print(df.columns)
df = df.sort_values(by="出现次数", ascending=False).reset_index(drop=True, names="序号")
df.index = pd.Index(range(1, len(df) + 1))

# 添加列
claimant_column = pd.Series([''] * (len(df) + 1))
df = df.assign(认领人=claimant_column)
pr_column = pd.Series([''] * (len(df) + 1))
df = df.assign(PR=pr_column)

with open(f'./classified_bytecode.md', 'w') as f:
    f.write(df.to_markdown())
