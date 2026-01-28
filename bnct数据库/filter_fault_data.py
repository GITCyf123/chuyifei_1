import pandas as pd
import numpy as np

# 读取清洗后的数据
df = pd.read_csv('cleaned_data.csv')

print("原始数据信息：")
print(f"总行数: {len(df)}")
print(f"总列数: {len(df.columns)}")
print(f"最后三列名称: {df.columns[-3:]}")

# 获取最后三列故障数据
fault_columns = df.columns[-3:]
print(f"\n故障列信息：")
for col in fault_columns:
    unique_values = df[col].unique()
    print(f"{col}: 唯一值 {unique_values}, 数据类型 {df[col].dtype}")

# 筛选最后三列都为0的行
df_filtered = df[(df[fault_columns[0]] == 0) & 
                 (df[fault_columns[1]] == 0) & 
                 (df[fault_columns[2]] == 0)]

print(f"\n筛选后的数据信息：")
print(f"筛选后行数: {len(df_filtered)}")
print(f"保留比例: {len(df_filtered) / len(df) * 100:.2f}%")

# 单独提取故障列数据
fault_data = df_filtered[fault_columns].copy()
print(f"\n提取的故障数据维度: {fault_data.shape}")
print(f"故障数据预览：")
print(fault_data.head())

# 统计筛选后的数据中故障列的值分布
print(f"\n筛选后故障列的值分布：")
for col in fault_columns:
    value_counts = fault_data[col].value_counts()
    print(f"{col}: {value_counts.to_dict()}")

# 保存筛选后的数据（包含所有列）
df_filtered.to_csv('filtered_all_data.csv', index=False, encoding='utf-8-sig')

# 保存只包含故障列的数据
fault_data.to_csv('fault_data_only.csv', index=False, encoding='utf-8-sig')

# 保存除了故障列以外的数据（用于模型训练）
non_fault_columns = df.columns[:-3]
df_no_fault = df_filtered[non_fault_columns].copy()
df_no_fault.to_csv('filtered_no_fault_data.csv', index=False, encoding='utf-8-sig')

print(f"\n文件保存完成：")
print(f"1. 筛选后包含所有列的数据: 'filtered_all_data.csv'")
print(f"2. 只包含故障列的数据: 'fault_data_only.csv'")
print(f"3. 不包含故障列的数据（可用于模型训练）: 'filtered_no_fault_data.csv'")

# 提供数据统计摘要
print(f"\n筛选后数据的统计摘要：")
print(f"\n关键电气参数统计（前5个）：")
electrical_cols = [col for col in df_filtered.columns if any(keyword in col for keyword in ['电压', '电流', '流强'])][:5]
for col in electrical_cols:
    min_val = df_filtered[col].min()
    max_val = df_filtered[col].max()
    mean_val = df_filtered[col].mean()
    std_val = df_filtered[col].std()
    print(f"{col}: 最小值={min_val:.6f}, 最大值={max_val:.6f}, 平均值={mean_val:.6f}, 标准差={std_val:.6f}")

# 检查是否有任何故障值
any_faults = (df_filtered[fault_columns] != 0).any().any()
print(f"\n验证筛选结果：")
print(f"筛选后的数据中是否包含任何故障值: {any_faults}")
if not any_faults:
    print("✓ 成功筛选出所有故障列为0的行")
else:
    print("! 警告：筛选后的数据中仍包含故障值")