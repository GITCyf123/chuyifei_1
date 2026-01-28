import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 检查CUDA是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 1. 加载筛选后的无故障数据
df = pd.read_csv('filtered_no_fault_data.csv')
print(f"加载的筛选后数据维度: {df.shape}")

# 2. 准备数据
# 选择特征和目标
# 电源相关参数作为主要特征
feature_cols = [col for col in df.columns if any(keyword in col for keyword in ['电压', '电流', '流强']) 
                and col != '等离子体电流']  # 排除目标变量

# 也可以加入一些温度参数
temp_cols = [col for col in df.columns if '温度' in col][:4]  # 选择前4个温度参数
feature_cols.extend(temp_cols)

target_col = '等离子体电流'  # 预测目标

print(f"\n使用的特征列 ({len(feature_cols)}个):")
for i, col in enumerate(feature_cols[:10], 1):  # 只显示前10个
    print(f"  {i}. {col}")
if len(feature_cols) > 10:
    print(f"  ... 等共{len(feature_cols)}个特征")
print(f"\n预测目标: {target_col}")

# 提取特征和目标
X = df[feature_cols].values
y = df[target_col].values

# 3. 数据预处理
# 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 4. 创建时间序列数据集
# iTransformer需要输入序列数据
def create_sequences(X, y, seq_length):
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X) - seq_length):
        X_seq = X[i:i+seq_length]
        y_seq = y[i+seq_length]  # 预测下一个时间步
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
    
    return np.array(X_sequences), np.array(y_sequences)

seq_length = 24  # 使用24个时间步作为历史数据
X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# 6. 转换为PyTorch张量并移至GPU
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32).to(device),
    torch.tensor(y_train, dtype=torch.float32).to(device)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32).to(device),
    torch.tensor(y_test, dtype=torch.float32).to(device)
)

# 数据加载器
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print(f"\n数据准备完成：")
print(f"训练集样本数: {len(train_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")
print(f"训练集批次: {len(train_loader)}")
print(f"测试集批次: {len(test_loader)}")
print(f"序列长度: {seq_length}")
print(f"特征维度: {X_seq.shape[2]}")

# 7. 实现iTransformer模型的简化版本
class iTransformer(nn.Module):
    def __init__(self, input_dim, seq_length, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(iTransformer, self).__init__()
        
        # 输入嵌入层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.position_encoding = nn.Parameter(torch.randn(1, seq_length, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        x = x + self.position_encoding
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 取最后一个时间步的输出
        x = x[:, -1, :]
        
        # 预测输出
        output = self.output_layer(x)
        
        return output.squeeze()

# 8. 初始化模型并移至GPU
input_dim = X_seq.shape[2]  # 特征维度
model = iTransformer(
    input_dim=input_dim,
    seq_length=seq_length,
    d_model=64,
    nhead=4,
    num_layers=2
).to(device)

# 9. 设置损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 10. 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        start_time = time.time()
        
        for X_batch, y_batch in train_loader:
            # 确保数据在正确的设备上
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * X_batch.size(0)
        
        # 计算平均损失
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # 打印训练进度
        end_time = time.time()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}, Time: {end_time-start_time:.2f}s')
    
    return train_losses

# 11. 评估函数
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # 确保数据在正确的设备上
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)
            
            # 将预测和实际值移回CPU以便numpy处理
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())
    
    # 计算平均损失
    test_loss /= len(test_loader.dataset)
    
    # 反标准化预测值和实际值
    predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
    
    return test_loss, predictions, actuals

# 12. 训练模型
print("\n开始训练模型...")
train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# 13. 评估模型
print("\n评估模型...")
test_loss, predictions, actuals = evaluate_model(model, test_loader, criterion)
print(f"测试集损失: {test_loss:.6f}")

# 14. 保存模型和标准化器
torch.save(model.state_dict(), 'itransformer_model_filtered.pth')

# 保存标准化器参数（用于推理）
np.save('scaler_X_mean.npy', scaler_X.mean_)
np.save('scaler_X_scale.npy', scaler_X.scale_)
np.save('scaler_y_mean.npy', scaler_y.mean_)
np.save('scaler_y_scale.npy', scaler_y.scale_)

print("\n模型和标准化器已保存：")
print("- 模型权重: itransformer_model_filtered.pth")
print("- 标准化器参数: scaler_X_mean.npy, scaler_X_scale.npy, scaler_y_mean.npy, scaler_y_scale.npy")

# 可视化结果
plt.figure(figsize=(15, 10))

# 绘制训练损失
plt.subplot(2, 1, 1)
plt.plot(train_losses, 'b-', linewidth=2)
plt.title('训练损失', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True, alpha=0.3)
plt.xlim(0, len(train_losses))

# 绘制预测vs实际值（前200个样本）
plt.subplot(2, 1, 2)
# 计算要显示的样本数量
n_samples = min(200, len(actuals))
sample_indices = np.arange(n_samples)

plt.plot(sample_indices, actuals[:n_samples], 'b-', linewidth=2, label='实际值', alpha=0.8)
plt.plot(sample_indices, predictions[:n_samples], 'r-', linewidth=1.5, label='预测值', alpha=0.8)

# 填充预测和实际值之间的区域
plt.fill_between(sample_indices, actuals[:n_samples], predictions[:n_samples],
                 alpha=0.3, color='gray', label='误差区域')

plt.title('等离子体电流预测 vs 实际值 (前{}个样本)'.format(n_samples), fontsize=14, fontweight='bold')
plt.xlabel('样本索引')
plt.ylabel('等离子体电流')
plt.legend()
plt.grid(True, alpha=0.3)

# 添加文本信息显示误差统计
if len(actuals) > 0:
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(mse)

    # 在图上添加误差统计
    textstr = f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
print("\n预测分析图已保存到 'prediction_analysis.png'")
plt.show()

# 如果需要额外的详细分析图，可以再创建一个图
plt.figure(figsize=(12, 8))
