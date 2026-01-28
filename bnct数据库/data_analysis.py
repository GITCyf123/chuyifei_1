import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('b20240604.csv')

print("数据基本信息：")
print(f"总行数: {len(df)}")
print(f"总列数: {len(df.columns)}")
print(f"\n前5行数据：")
print(df.head())

# 找出所有值都相同的列
constant_columns = []
for col in df.columns:
    if df[col].nunique() == 1:
        constant_columns.append(col)

print(f"\n所有值都相同的列 ({len(constant_columns)}个):")
for i, col in enumerate(constant_columns, 1):
    print(f"{i}. {col}: {df[col].iloc[0]}")

# 删除所有值都相同的列
df_cleaned = df.drop(columns=constant_columns)
print(f"\n删除常量列后的数据维度: {df_cleaned.shape}")

# 分析剩余的列
print(f"\n剩余列列表 ({len(df_cleaned.columns)}个):")
for i, col in enumerate(df_cleaned.columns, 1):
    print(f"{i}. {col}")

# 分析每列的数据类型和统计信息
print(f"\n数据统计信息：")
print(df_cleaned.describe())

# 对列进行分类，根据列名和数据特征
# 电源相关参数
electrical_cols = [col for col in df_cleaned.columns if any(keyword in col for keyword in ['电压', '电流', '流强'])]

# 温度相关参数
temp_cols = [col for col in df_cleaned.columns if '温度' in col or col.startswith('T_')]

# 时间列
time_cols = [col for col in df_cleaned.columns if '时间' in col]

# 位置和其他参数
other_cols = [col for col in df_cleaned.columns if col not in electrical_cols + temp_cols + time_cols]

print(f"\n列分类结果：")
print(f"电源相关参数 ({len(electrical_cols)}个): {electrical_cols}")
print(f"温度相关参数 ({len(temp_cols)}个): {temp_cols}")
print(f"时间列 ({len(time_cols)}个): {time_cols}")
print(f"其他参数 ({len(other_cols)}个): {other_cols}")

# 建议的特征值和输出值（基于iTransformer模型的时间序列预测特性）
print(f"\niTransformer模型建议：")
print(f"特征值候选（可以根据具体预测目标调整）：")
print(f"1. 历史时序特征：所有电气参数和部分温度参数")
print(f"2. 时间特征：从时间列中提取的小时、分钟等信息")

print(f"\n输出值候选（根据监控重点选择）：")
print(f"1. 关键电气参数：如等离子体电压、等离子体电流")
print(f"2. 关键温度参数：如室内机出水温度、Bnct靶水机进水温度")
print(f"3. 系统状态参数：如各种流强参数")

# 保存清洗后的数据
df_cleaned.to_csv('cleaned_data.csv', index=False, encoding='utf-8-sig')
print(f"\n清洗后的数据已保存到 'cleaned_data.csv'")

# 生成一个简单的iTransformer数据准备示例
print(f"\n生成iTransformer数据准备示例...")

# 假设我们选择等离子体电流作为预测目标
target_col = '等离子体电流'  # 可能需要根据实际列名调整
if target_col in df_cleaned.columns:
    # 提取特征和目标
    feature_cols = [col for col in electrical_cols if col != target_col]
    X = df_cleaned[feature_cols].values
    y = df_cleaned[target_col].values
    
    print(f"示例特征列数: {len(feature_cols)}")
    print(f"示例特征形状: {X.shape}")
    print(f"示例目标形状: {y.shape}")
    
    # 简单的时间序列分割示例（用于iTransformer）
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i+seq_length])
        return np.array(sequences)
    
    # 创建序列数据（示例参数）
    seq_length = 10
    if len(df_cleaned) > seq_length:
        # 创建特征序列
        X_sequences = create_sequences(X, seq_length)
        # 创建目标序列（预测未来1个时间步）
        y_sequences = y[seq_length:]
        
        print(f"\n创建的序列数据形状：")
        print(f"特征序列形状: {X_sequences.shape}")  # (样本数, 序列长度, 特征数)
        print(f"目标序列形状: {y_sequences.shape}")  # (样本数,)
        print(f"\n这是iTransformer模型的典型输入格式")
    else:
        print("数据长度不足，无法创建序列")
else:
    print(f"目标列 '{target_col}' 不存在于数据中，请调整目标列名")