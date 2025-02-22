import cv2
import numpy as np
from PIL import Image

def resize_and_save_1080_with_dpi(image_path, save_path):
    # 读取图片
    image = Image.open(image_path)
    
    # Convert RGBA to RGB if necessary
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # 获取原始尺寸
    width, height = image.size
    
    # 计算新的宽度，保持比例
    new_width = int(width * (720 / height))
    
    # 调整尺寸
    resized_image = image.resize((new_width, 720), Image.Resampling.LANCZOS)
    
    # 设置72 DPI (72像素/英寸)
    resized_image.info['dpi'] = (72, 72)
    
    # 保存图片
    resized_image.save(save_path, dpi=(72, 72))
    return resized_image

resize_and_save_1080_with_dpi("data/77.jpg", "data/77_r.jpg")
