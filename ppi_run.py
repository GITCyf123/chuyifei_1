import os
from matplotlib.colors import ListedColormap
from PPI_Data import SequenceDataset
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model import Res50_model, ResNet50WithAttention, Inception
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE

epochs = 1
lookback_len = 3
batch_size = 8
directory = 'D:\\py\\LearnPython\\挑杯\\挑战杯训练数据集\\地杂波训练数据集'
# def get_all_data(directory):
#     all_data = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.npz'):
#             file_path = os.path.join(directory, filename)
#             data = SequenceDataset(file_path, lookback_len)
#             all_data.append(data)
#     all_data = np.concatenate(all_data, axis=0)
#     print(all_data.shape)
#     return all_data

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')


    model = Res50_model.ResNet().to(device)
    #model = ResNet50WithAttention.ResNet50WithAttention().to(device)
    #model = Inception.InceptionWithAttention().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()  # BCELoss
    model.train()

    for filename in os.listdir(directory):

        if filename.endswith('.npz'):
            file_path = os.path.join(directory, filename)
            
            file_path = 'D:\\py\\LearnPython\\挑杯\\挑战杯训练数据集\\地杂波训练数据集\\Z1_20230826000131.npz'
            filename = 'Z1_20230826000131.npz'
            dataset = SequenceDataset(file_path, lookback_len)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            for epoch in range(epochs):
                total_loss = 0.0

                all_features = []
                all_labels = []
                for batch_idx, data in enumerate(dataloader):
                    # [num_points, lookback_len, lookback_len, num_channels]
                    data = data.to(device)
                    optimizer.zero_grad()
                    f_out, output = model(data)
                    prob_out = F.softmax(output, dim=1)
                    _, predictions = torch.max(prob_out, dim=1)

                    # fea = np.concatenate((f_out.detach().cpu().numpy(), h_f_out.detach().cpu().numpy()), axis=1)
                    # all_features.append(fea)
                    all_features.append(f_out.detach().cpu().numpy())
                    all_labels.append(predictions.detach().cpu().numpy())

                all_features_np = np.concatenate(all_features, axis=0)  # [num_samples, feature_dim]
                all_labels_np = np.concatenate(all_labels, axis=0)  # [num_samples, 1]

                kmeans = KMeans(n_clusters=2, random_state=0).fit(all_features_np)
                pseudo_labels = kmeans.labels_

                # 确定标签

                # 计算伪标签和模型分类结果之间的损失，并进行梯度回传
                pseudo_labels_tensor = torch.tensor(pseudo_labels, dtype=torch.long).to(device)
                # 将伪标签转换为 [num_samples, 1]

                for batch_idx, data in enumerate(dataloader):
                    data = data.to(device)
                    batch_start = batch_idx * batch_size
                    batch_end = batch_start + batch_size
                    pseudo_labels_batch = pseudo_labels_tensor[batch_start:batch_end]

                    optimizer.zero_grad()
                    f_out, output = model(data)
                    # print("output====")
                    # print(output)
                    # print("pseudo_labels_batch====")
                    # print(pseudo_labels_batch)
                    loss = criterion(output, pseudo_labels_batch)
                    # print('loss===')
                    # print(loss)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')

            
            
            # 分类结果可视化
            data = np.load(file_path)
            Z1 = data['Z1']
            V1 = data['V1']
            W1 = data['W1']
            SNR1 = data['SNR1']
            # LDR = data['LDR']

            classification_results = np.zeros_like(Z1, dtype=np.float32)

            all_labels = []
            all_features = []
            prob = []
            for batch_idx, data in enumerate(dataloader):
                data = data.to(device)
                # print("data====")
                # print(data)
                f_out, output = model(data)
                # print('output====')
                # print(output)
                batch_size = data.shape[0]
                prob_out = F.softmax(output, dim=1)
                predictions = torch.argmax(prob_out, dim=1).cpu().numpy()
                # for p in prob_out:
                #     prob.append(p.detach().cpu().numpy())
                all_features.append(f_out.detach().cpu().numpy())
                all_labels.append(predictions)
            all_features_np = np.concatenate(all_features, axis=0)
            all_labels_np = np.concatenate(all_labels, axis=0)
            # np.savetxt('prob.txt',prob, delimiter=',')
            # kmeans
            # kmeans = KMeans(n_clusters=3, random_state=0).fit(all_features_np)
            # all_labels_np = kmeans.labels_
            id=0
            for i in range(Z1.shape[0]):
                for j in range(Z1.shape[1]):
                    z1_not_nan = not np.isnan(Z1[i, j])
                    v1_not_nan = not np.isnan(V1[i, j])
                    w1_not_nan = not np.isnan(W1[i, j])
                    snr1_not_nan = not np.isnan(SNR1[i, j])
                    # ldr_not_nan = not np.isnan(Z1[i, j])
                    if z1_not_nan or v1_not_nan or w1_not_nan or snr1_not_nan:
                        classification_results[i,j] = all_labels_np[id]
                        id += 1
                    else:
                        classification_results[i, j] = 2

            print("可视化结果")
            # 执行K-Means聚类
            kmeans = KMeans(n_clusters=2, random_state=0)
            y_kmeans = kmeans.fit_predict(all_features_np)
            
            # # 使用t-SNE进行降维
            # tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
            # X_tsne = tsne.fit_transform(all_features_np)

            # # 可视化聚类结果
            # plt.figure(figsize=(10, 7))
            # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_kmeans, cmap='viridis')
            # plt.title('t-SNE Visualization of K-Means Clustering')
            # plt.xlabel('t-SNE 1')
            # plt.ylabel('t-SNE 2')
            # plt.colorbar()
            # plt.show()
            
            
            # # 嵌入3维
            # tsne = TSNE(n_components=3, perplexity=30, n_iter=300)
            # X_tsne = tsne.fit_transform(all_features_np)

            # # 可视化聚类结果
            # fig = plt.figure(figsize=(10, 7))
            # ax = fig.add_subplot(111, projection='3d')
            # scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y_kmeans, cmap='viridis')

            # ax.set_title('3D t-SNE Visualization of K-Means Clustering')
            # ax.set_xlabel('t-SNE 1')
            # ax.set_ylabel('t-SNE 2')
            # ax.set_zlabel('t-SNE 3')
            # fig.colorbar(scatter)
            # plt.show()
            
            cmap = ListedColormap(['blue', 'red','white'])
            data = classification_results[:,:]
            print(np.unique(data))
            
            data= data.astype(int)
            # 
            print("各类别数量")
            print(np.count_nonzero(data ==0))
            print(np.count_nonzero(data ==1))
            print(np.count_nonzero(data ==2))
            # 生成角度和半径
            angles = np.linspace(0, 2 * np.pi, 360)
            radii = np.arange(500)

            # 创建极坐标图
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
            # 将0°放置在最上方
            ax.set_theta_offset(np.pi / 2)
            # 使角度顺时针增加
            ax.set_theta_direction(-1)
            # 将数据映射到极坐标上
            angle_grid, radius_grid = np.meshgrid(angles, radii, indexing='ij')

            # 绘制极坐标图
            c = ax.pcolormesh(angle_grid, radius_grid, data, shading='auto', cmap=cmap) #'viridis'

            # 添加颜色条
            fig.colorbar(c, ax=ax)

            plt.title('Polar Plot of Colors')
            plt.show()
            plt.savefig('polar_plot.png')
        
        # 保存结果
        torch.save(model, 'C:/Users/DELL/Desktop/挑战杯/2code/ppi/pure_p2s_model.pkl')
        print("模型保存成功")
