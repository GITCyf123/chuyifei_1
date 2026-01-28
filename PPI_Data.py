import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SequenceDataset:
    # lookback_len: 需要为奇数，提取以点为中心的边长为lookback_len的正方形区域的特征序列
    def __init__(self, npz_file,lookback_len):
        # 加载 .npz 文件
        data = np.load(npz_file)

        # 提取数据
        self.Z1 = data['Z1']
        self.V1 = data['V1']
        self.W1 = data['W1']
        self.SNR1 = data['SNR1']
        # print(f"Z1 shape: {self.Z1.shape}")
        # 记录非 NAN 值和坐标
        self.non_nan_values_with_coords = []
        for i in range(self.Z1.shape[0]):
            for j in range(self.Z1.shape[1]):
                z1_not_nan = not np.isnan(self.Z1[i, j])
                v1_not_nan = not np.isnan(self.V1[i, j])
                w1_not_nan = not np.isnan(self.W1[i, j])
                snr1_not_nan = not np.isnan(self.SNR1[i, j])
                if z1_not_nan or v1_not_nan or w1_not_nan or snr1_not_nan:
                    self.non_nan_values_with_coords.append((i, j))
        # print(f"统计的非NAN值共有：{len(self.non_nan_values_with_coords)}")
        self.lookback_len = lookback_len
        self.sequences = self.get_sequences_around_non_nan()
        print(self.sequences.shape)

        # 将 sequences 张量移动到设备
        self.sequences = self.sequences.to(device)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index].to(device)

    #  提取以每个非 NAN 值为中心的径向长度为lookback_len的序列，返回张量序列，ret = [num_points, lookback_len, channels]
    def get_sequences_around_non_nan(self):
        sequences = []
        sequence_list = []
        half_len = (self.lookback_len-1) // 2
        remain = (self.lookback_len-1)%2
        for (row, col) in self.non_nan_values_with_coords:
            # 确定正方形矩阵的左上角和右下角坐标 360*500
            # start_row = max(0, row - half_len)
            # end_row = min(self.Z1.shape[0], row + half_len + remain)
            start_col = max(0, col - half_len)
            end_col = min(self.Z1.shape[1], col + half_len + 1)            

            # 计算目标区域的左上角和右下角坐标
            # target_start_row = max(0, half_len - row)
            # target_end_row = target_start_row + (end_row - start_row)
            target_start_col = max(0, half_len - col)
            target_end_col = target_start_col + (end_col - start_col)

            # 选取的角度，360和0为一个方向，不断开
            select_angle = list((np.arange(row - half_len, row + half_len + remain +1)+360)%360)
            
            # 创造0矩阵
            z1_sequence=np.zeros((self.lookback_len, self.lookback_len), dtype=self.Z1.dtype)
            v1_sequence=np.zeros((self.lookback_len, self.lookback_len), dtype=self.V1.dtype)
            w1_sequence=np.zeros((self.lookback_len, self.lookback_len), dtype=self.W1.dtype)
            snr1_sequence=np.zeros((self.lookback_len, self.lookback_len), dtype=self.SNR1.dtype)
            # # 加上坐标
            # angle_sequence=np.full((self.lookback_len, self.lookback_len), row)
            # radius_sequence=np.full((self.lookback_len, self.lookback_len), col)
            
            # 将row，col为中心的矩阵复制
            # z1_sequence[target_start_row:target_end_row, target_start_col:target_end_col] = self.Z1[start_row:end_row, start_col:end_col]
            # v1_sequence[target_start_row:target_end_row, target_start_col:target_end_col] = self.V1[start_row:end_row, start_col:end_col]
            # w1_sequence[target_start_row:target_end_row, target_start_col:target_end_col] = self.W1[start_row:end_row, start_col:end_col]
            # snr1_sequence[target_start_row:target_end_row, target_start_col:target_end_col] = self.SNR1[start_row:end_row, start_col:end_col]

            # 复制
            z1_sequence[:, target_start_col:target_end_col] = self.Z1[select_angle, start_col:end_col]
            v1_sequence[:, target_start_col:target_end_col] = self.V1[select_angle, start_col:end_col]
            w1_sequence[:, target_start_col:target_end_col] = self.W1[select_angle, start_col:end_col]
            snr1_sequence[:, target_start_col:target_end_col] = self.SNR1[select_angle, start_col:end_col]

            # 使用 np.nan_to_num 将 NaN 值替换为 0
            z1_sequence = np.nan_to_num(z1_sequence, nan=0.0)
            v1_sequence = np.nan_to_num(v1_sequence, nan=0.0)
            w1_sequence = np.nan_to_num(w1_sequence, nan=0.0)
            snr1_sequence = np.nan_to_num(snr1_sequence, nan=0.0)
            # ldr_sequence = np.nan_to_num(ldr_sequence, nan=0.0)

            # # 不够的用0进行填充
            # z1_sequence = np.pad(z1_sequence, (0, self.lookback_len - len(z1_sequence)), mode='constant', constant_values=0)
            # v1_sequence = np.pad(v1_sequence, (0, self.lookback_len - len(v1_sequence)), mode='constant', constant_values=0)
            # w1_sequence = np.pad(w1_sequence, (0, self.lookback_len - len(w1_sequence)), mode='constant', constant_values=0)
            # snr1_sequence = np.pad(snr1_sequence, (0, self.lookback_len - len(snr1_sequence)), mode='constant', constant_values=0)
            # ldr_sequence = np.pad(ldr_sequence, (0, self.lookback_len - len(ldr_sequence)), mode='constant', constant_values=0)

            # 将序列转换为张量
            z1_tensor = torch.tensor(z1_sequence, dtype=torch.float32).unsqueeze(2)
            v1_tensor = torch.tensor(v1_sequence, dtype=torch.float32).unsqueeze(2)
            w1_tensor = torch.tensor(w1_sequence, dtype=torch.float32).unsqueeze(2)
            snr1_tensor = torch.tensor(snr1_sequence, dtype=torch.float32).unsqueeze(2)
            
            # angle_tensor = torch.tensor(angle_sequence, dtype=torch.float32).unsqueeze(2)
            # radius_tensor = torch.tensor(radius_sequence, dtype=torch.float32).unsqueeze(2)
            # print(z1_sequence.shape)
            # ldr_tensor = torch.tensor(ldr_sequence, dtype=torch.float32).view(-1, 1)

            combined_tensor = torch.cat([z1_tensor, v1_tensor, w1_tensor, snr1_tensor], dim=2)
            # print(combined_tensor.shape)
            # 将序列添加到列表中，每个元素是一个包含五个序列的元组
            sequence_list.append(combined_tensor)

        # 将二维张量堆叠成4维张量
        sequences = torch.stack(sequence_list, dim=0)  # [num_points, lookback_len,lookback_len, channels]
        return sequences

# if __name__ == "__main__":
#     npz_file = 'C:/Users/DELL/Desktop/挑战杯/挑战杯训练数据集/地杂波训练数据集/Z1_20230826000131.npz'
#     lookback_len = 10
#     print(f"使用的设备: {device}")
#     dataset = SequenceDataset(npz_file, lookback_len)

