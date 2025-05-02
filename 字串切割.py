import os
import cv2
import numpy as np
from PIL import Image
import random
import shutil
from tqdm import tqdm

def create_output_dirs(base_output_dir):
    """創建輸出目錄"""
    # 如果目錄已存在，先刪除
    if os.path.exists(base_output_dir):
        shutil.rmtree(base_output_dir)
    os.makedirs(base_output_dir)
    
    # 創建每個字符的子目錄
    chars = ['2', '3', '4', '5', '6', '7', '8', 'a', 'b', 'c', 'd', 
             'e', 'f', 'g', 'h', 'k', 'm', 'p', 'r', 'w', 'x', 'y']
    for char in chars:
        os.makedirs(os.path.join(base_output_dir, char), exist_ok=True)

def preprocess_image(image):
    """預處理圖片"""
    # 轉換為灰度圖
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 使用形態學操作去除細小的干擾線
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def split_and_save_chars(input_dir, output_dir):
    """切分並保存字符圖片"""
    # 獲取所有圖片文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    counter = 0
    
    # 使用tqdm顯示進度
    for img_file in tqdm(image_files, desc="Processing images"):
        # 讀取圖片和標籤
        img_path = os.path.join(input_dir, img_file)
        label = os.path.splitext(img_file)[0]
        image = Image.open(img_path).convert('RGB')
        width = image.width
        char_width = width // 6
        
        # 為每個字符創建切分圖片
        for i, char in enumerate(label):
            # 計算切分區域
            start_x = i * char_width
            end_x = start_x + char_width
            
            # 切分字符
            char_image = image.crop((start_x, 0, end_x, image.height))
            
            # 生成新的文件名
            new_filename = f"{char}_{counter:05d}.png"
            save_path = os.path.join(output_dir, char, new_filename)
            
            # 保存圖片
            char_image.save(save_path)
            counter += 1
            
        # 每處理100張圖片輸出一次進度
        if counter % 600 == 0:
            print(f"Processed {counter // 6} images, created {counter} character images")

def verify_split_results(output_dir):
    """驗證切分結果"""
    total_chars = 0
    chars_count = {}
    
    # 統計每個字符的數量
    for char_dir in os.listdir(output_dir):
        dir_path = os.path.join(output_dir, char_dir)
        if os.path.isdir(dir_path):
            count = len(os.listdir(dir_path))
            chars_count[char_dir] = count
            total_chars += count
    
    # 輸出統計信息
    print("\n切分結果統計:")
    print(f"總字符數: {total_chars}")
    print("\n每個字符的數量:")
    for char, count in sorted(chars_count.items()):
        print(f"字符 '{char}': {count} 張")

def main():
    # 設置輸入和輸出目錄
    input_dir = r"C:\Users\C14511\Desktop\captcha\CAPTCHA\training" # 請修改為您的輸入目錄
    output_dir = r"C:\Users\C14511\Desktop\captcha\CAPTCHA\abcd"  # 請修改為您的輸出目錄
    
    # 創建輸出目錄結構
    create_output_dirs(output_dir)
    
    # 執行切分和保存
    try:
        split_and_save_chars(input_dir, output_dir)
        print("\n圖片切分完成!")
        
        # 驗證結果
        verify_split_results(output_dir)
        
    except Exception as e:
        print(f"處理過程中發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()