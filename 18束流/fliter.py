import cv2
import numpy as np

# 读取图片
image = cv2.imread('data.png')

if image is None:
    print("无法读取图片，请检查data.png文件是否存在")
else:
    # 转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    max = 0
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            if(max<gray_image[i][j]):
                max = gray_image[i][j]
    # 输出灰度矩阵信息
    print("灰度矩阵的形状:", gray_image.shape)
    print("灰度矩阵的数据类型:", gray_image.dtype)
    
    # 设置numpy输出选项，显示完整矩阵
    np.set_printoptions(threshold=np.inf, linewidth=200)
    print("灰度矩阵:")
    print(gray_image)

    
    # 高亮灰度值超过200的像素
    highlighted_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # 转换为BGR以便显示颜色
    
    # 找到灰度值超过200的像素位置
    high_value_pixels = np.where(gray_image > 160)
    
    # 将这些像素标记为红色
    highlighted_image[high_value_pixels] = [255, 200, 100]  # BGR格式，红色
    
    # 保存高亮图片
    cv2.imwrite('highlighted_high_values.png', highlighted_image)
    print(f"高亮图片已保存为 highlighted_high_values.png")
    print(f"灰度值超过200的像素数量: {len(high_value_pixels[0])}")
    
    # 保存灰度图
    cv2.imwrite('gray_data.png', gray_image)
    print("灰度图已保存为 gray_data.png")