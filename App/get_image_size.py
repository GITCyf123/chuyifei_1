from PIL import Image
import os

# 图片路径
img_path = 'jpg/9_5main.jpg'

# 确保图片存在
if os.path.exists(img_path):
    # 打开图片
    with Image.open(img_path) as img:
        # 获取图片尺寸
        width, height = img.size
        print(f"图片宽度: {width}px")
        print(f"图片高度: {height}px")
        # 计算宽高比
        aspect_ratio = width / height
        print(f"图片宽高比: {aspect_ratio:.2f}")
else:
    print(f"图片不存在: {img_path}")