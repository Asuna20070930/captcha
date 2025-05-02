import os
import cv2
import numpy as np

def remove_horizontal_lines(image):
    # 複製原始圖像
    cleaned = image.copy()
    
    # 創建水平線檢測核心
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image.shape[1], 1))
    
    # 檢測水平線
    horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # 找出輪廓
    contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 遮罩
    mask = np.ones(image.shape, dtype=np.uint8) * 255
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 精確判斷橫線
        if w > image.shape[1] * 0.4 and h < 3:
            # 將橫線區域設為黑色
            cv2.rectangle(mask, (0, max(0, y-1)), (image.shape[1], min(image.shape[0], y+h+1)), 0, -1)
    
    # 應用遮罩
    cleaned = cv2.bitwise_and(cleaned, mask)
    
    return cleaned

def process_captcha_images(input_folder, output_folder):
    # 確保輸出資料夾存在，並清除舊檔
    os.makedirs(output_folder, exist_ok=True)
    for old_file in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, old_file))

    # 遍歷輸入資料夾中的所有圖片
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # 讀取圖片
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # 高斯模糊
            blurred = cv2.GaussianBlur(image, (3, 3), 0)

            # 自適應二值化
            binary = cv2.adaptiveThreshold(
                blurred, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                11, 
                2
            )

            # 去除橫線
            cleaned = remove_horizontal_lines(binary)

            # 形態學操作：輕微去除雜點
            kernel = np.ones((3,3), np.uint8)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

            # 儲存處理後的圖片
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cleaned)

    print("圖片處理完成！")

# 設定輸入和輸出資料夾路徑
input_folder = r"C:\Users\C14511\Desktop\captcha\CAPTCHA\000"
output_folder = r"C:\Users\C14511\Desktop\captcha\CAPTCHA\0000"

# 執行圖片處理1
process_captcha_images(input_folder, output_folder)