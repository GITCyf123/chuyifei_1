import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_brightest_spot_edge(image_path):
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片 {image_path}")
        return
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 找到图像中亮度最高的像素值
    max_brightness = np.max(gray)
    print(f"图像最大亮度值: {max_brightness}")
    
    # 创建二值掩码，只保留亮度值接近最大值的像素
    # 设置阈值为最大亮度值的95%以上
    threshold = max_brightness * 0.7
    _, binary_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # 使用形态学操作清理掩码
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # 找到连通区域
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("未找到光斑")
        return
    
    # 找到最大的连通区域（最亮的光斑）
    largest_contour = max(contours, key=cv2.contourArea)
    print(f"最大光斑面积: {cv2.contourArea(largest_contour)} 像素")
    
    # 创建边缘检测结果图像
    edge_result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # 在原图上绘制边缘
    cv2.drawContours(edge_result, [largest_contour], -1, (255, 200, 100), 2)  # 绿色边缘
    
    # 计算光斑的中心
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # 在中心点画一个标记
        cv2.circle(edge_result, (cx, cy), 5, (255, 0, 0), -1)  # 蓝色中心点
        print(f"光斑中心位置: ({cx}, {cy})")
    
    # 使用Canny边缘检测作为对比
    print(f"灰度图像的最小值: {np.min(gray)}, 最大值: {np.max(gray)}")
    
    # 尝试不同的Canny阈值参数
    edges_canny = cv2.Canny(gray, 50, 150)
    print(f"Canny边缘检测结果 - 非零像素数: {np.count_nonzero(edges_canny)}")
    
    # 如果标准Canny没有检测到边缘，尝试自适应阈值
    if np.count_nonzero(edges_canny) < 100:
        print("尝试自适应Canny边缘检测...")
        median = np.median(gray)
        lower = int(max(0, 0.7 * median))
        upper = int(min(255, 1.3 * median))
        edges_canny = cv2.Canny(gray, lower, upper)
        print(f"自适应Canny边缘检测结果 - 非零像素数: {np.count_nonzero(edges_canny)}")
    
    # 保存原始Canny结果用于调试
    cv2.imwrite('canny_raw.png', edges_canny)
    
    # 只保留Canny边缘中属于最亮光斑的部分
    mask_3d = np.stack([binary_mask] * 3, axis=-1)
    edges_canny_colored = cv2.cvtColor(edges_canny, cv2.COLOR_GRAY2BGR)
    edges_canny_colored = cv2.bitwise_and(edges_canny_colored, mask_3d)
    
    # 保存结果
    cv2.imwrite('brightest_spot_edge.png', edge_result)
    cv2.imwrite('binary_mask.png', binary_mask)
    cv2.imwrite('canny_edges_brightest.png', edges_canny_colored)
    
    print("结果已保存:")
    print("- brightest_spot_edge.png: 检测到的光斑边缘（绿色）和中心（蓝色）")
    print("- binary_mask.png: 二值化掩码")
    print("- canny_edges_brightest.png: Canny边缘检测结果")
    
    return edge_result, binary_mask, edges_canny_colored

if __name__ == "__main__":
    # 检测data.png中最亮光斑的边缘
    result = detect_brightest_spot_edge('data.png')