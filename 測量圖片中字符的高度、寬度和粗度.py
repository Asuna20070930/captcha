# 載入需要的函式庫
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
from collections import defaultdict

def analyze_captcha_characters(image_path):
    """分析驗證碼圖片中各個字符的特性"""
    # 讀取圖片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"無法讀取圖片: {image_path}")
        return None
    
    # 二值化處理，將灰度圖轉為黑白圖
    _, binary = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY_INV)  # 調整閾值以適應灰度驗證碼
    
    # 查找連通區域（即字符）
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 按從左到右的順序排序連通區域
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    
    # 獲取檔名作為答案（去掉路徑和副檔名）
    filename = os.path.basename(image_path)
    answer = os.path.splitext(filename)[0]
    
    # 提取實際的字符（排除字體標記）
    captcha_text = ""
    for i in range(0, len(answer), 2):
        if i + 1 < len(answer):
            captcha_text += answer[i+1]
    
    # 如果檔名本身就是字符（長度為6），則直接使用
    if len(answer) == 6:
        captcha_text = answer
    
    # 確保找到的連通區域數量與字符數量匹配
    if len(contours) != len(captcha_text) and len(contours) != 7:  # 考慮到干擾線可能被識別為一個額外的連通區域
        print(f"警告: 在圖片 {filename} 中找到 {len(contours)} 個連通區域，但答案有 {len(captcha_text)} 個字符")
        # 繪製找到的連通區域，用於診斷
        debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(f"debug_{filename}", debug_img)
        
        # 僅在相差不大時繼續
        if abs(len(contours) - len(captcha_text)) > 2:
            return None
    
    # 分析每個連通區域的特性
    char_properties = []
    valid_contours = []
    
    # 過濾掉可能的噪點和干擾線
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # 過濾掉太小的連通區域（可能是噪點）和太窄的連通區域（可能是干擾線的一部分）
        if area > 50 and w > 5 and h > 10:  # 根據您的驗證碼調整這些閾值
            valid_contours.append(contour)
    
    # 如果過濾後的連通區域數量與字符數量匹配，則繼續分析
    if len(valid_contours) == len(captcha_text):
        for i, contour in enumerate(valid_contours):
            char = captcha_text[i] if i < len(captcha_text) else "?"
            
            # 獲取外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 計算粗度（使用連通區域的面積除以高度作為近似值）
            area = cv2.contourArea(contour)
            thickness = area / h if h > 0 else 0
            
            # 將字符特性添加到列表
            char_properties.append({
                "char": char,
                "width": w,
                "height": h,
                "thickness": thickness,
                "area": area,
                "x": x,
                "y": y
            })
    
    return char_properties

def batch_analyze_captchas(directory):
    """批量分析目錄中的所有驗證碼圖片"""
    # 確保目錄存在
    if not os.path.exists(directory):
        print(f"目錄不存在: {directory}")
        return None
    
    # 獲取目錄中所有PNG圖片
    image_files = [f for f in os.listdir(directory) if f.lower().endswith('.png')]
    
    if not image_files:
        print(f"目錄中沒有找到PNG圖片: {directory}")
        return None
    
    print(f"找到 {len(image_files)} 張PNG圖片進行分析")
    
    # 收集所有字符的特性
    all_char_properties = {}
    char_counts = defaultdict(int)
    
    for i, img_file in enumerate(image_files):
        if i % 10 == 0:
            print(f"正在處理: {i+1}/{len(image_files)}")
        
        img_path = os.path.join(directory, img_file)
        properties = analyze_captcha_characters(img_path)
        
        if properties:
            for prop in properties:
                char = prop["char"]
                char_counts[char] += 1
                
                if char not in all_char_properties:
                    all_char_properties[char] = {
                        "widths": [],
                        "heights": [],
                        "thicknesses": [],
                        "areas": []
                    }
                
                all_char_properties[char]["widths"].append(prop["width"])
                all_char_properties[char]["heights"].append(prop["height"])
                all_char_properties[char]["thicknesses"].append(prop["thickness"])
                all_char_properties[char]["areas"].append(prop["area"])
    
    # 計算每個字符的平均特性
    char_stats = {}
    for char, props in all_char_properties.items():
        char_stats[char] = {
            "count": char_counts[char],
            "avg_width": np.mean(props["widths"]),
            "avg_height": np.mean(props["heights"]),
            "avg_thickness": np.mean(props["thicknesses"]),
            "avg_area": np.mean(props["areas"]),
            "std_width": np.std(props["widths"]),
            "std_height": np.std(props["heights"]),
            "std_thickness": np.std(props["thicknesses"])
        }
    
    return char_stats

def visualize_char_stats(char_stats):
    """可視化字符統計資料"""
    if not char_stats:
        print("沒有統計資料可以可視化")
        return
    
    # 準備數據
    chars = list(char_stats.keys())
    heights = [stats["avg_height"] for char, stats in char_stats.items()]
    widths = [stats["avg_width"] for char, stats in char_stats.items()]
    thicknesses = [stats["avg_thickness"] for char, stats in char_stats.items()]
    
    # 按照字符排序
    sorted_indices = np.argsort(chars)
    chars = [chars[i] for i in sorted_indices]
    heights = [heights[i] for i in sorted_indices]
    widths = [widths[i] for i in sorted_indices]
    thicknesses = [thicknesses[i] for i in sorted_indices]
    
    # 創建圖表
    plt.figure(figsize=(15, 10))
    
    # 高度條形圖
    plt.subplot(3, 1, 1)
    plt.bar(chars, heights, color='blue')
    plt.title('字符平均高度')
    plt.ylabel('像素')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 寬度條形圖
    plt.subplot(3, 1, 2)
    plt.bar(chars, widths, color='green')
    plt.title('字符平均寬度')
    plt.ylabel('像素')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 粗度條形圖
    plt.subplot(3, 1, 3)
    plt.bar(chars, thicknesses, color='red')
    plt.title('字符平均粗度 (面積/高度)')
    plt.ylabel('像素')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('char_stats.png')
    plt.show()
    
    # 輸出表格形式的數據
    print("\n字符統計表格:")
    print("字符\t計數\t平均高度\t平均寬度\t平均粗度\t平均面積")
    print("-" * 70)
    
    for char in chars:
        stats = char_stats[char]
        print(f"{char}\t{stats['count']}\t{stats['avg_height']:.1f}\t{stats['avg_width']:.1f}\t"
              f"{stats['avg_thickness']:.1f}\t{stats['avg_area']:.1f}")
    
    # 分析建議
    print("\n分析建議:")
    all_heights = [stats["avg_height"] for stats in char_stats.values()]
    all_widths = [stats["avg_width"] for stats in char_stats.values()]
    all_thicknesses = [stats["avg_thickness"] for stats in char_stats.values()]
    
    # 根據字體判斷字符集
    arial_chars = '68bcdefghmpwxy'
    ebrima_chars = '23457ak'
    
    # 計算各組字符的平均高度和粗度
    arial_heights = [char_stats[c]["avg_height"] for c in char_stats if c in arial_chars]
    ebrima_heights = [char_stats[c]["avg_height"] for c in char_stats if c in ebrima_chars]
    
    arial_thicknesses = [char_stats[c]["avg_thickness"] for c in char_stats if c in arial_chars]
    ebrima_thicknesses = [char_stats[c]["avg_thickness"] for c in char_stats if c in ebrima_chars]
    
    if arial_heights and ebrima_heights:
        avg_arial_height = np.mean(arial_heights)
        avg_ebrima_height = np.mean(ebrima_heights)
        
        avg_arial_thickness = np.mean(arial_thicknesses)
        avg_ebrima_thickness = np.mean(ebrima_thicknesses)
        
        print(f"Arial字符平均高度: {avg_arial_height:.1f}像素")
        print(f"Ebrima字符平均高度: {avg_ebrima_height:.1f}像素")
        
        print(f"Arial字符平均粗度: {avg_arial_thickness:.1f}像素")
        print(f"Ebrima字符平均粗度: {avg_ebrima_thickness:.1f}像素")
        
        # 建議調整
        height_diff = avg_ebrima_height - avg_arial_height
        thickness_diff = avg_ebrima_thickness - avg_arial_thickness
        
        print("\n基於分析的調整建議:")
        
        # 建議字體大小調整
        if abs(height_diff) > 3:
            if height_diff > 0:
                print(f"建議增加Arial字體大小約 {int(height_diff / 2)}像素，或減少Ebrima字體大小約 {int(height_diff / 2)}像素")
            else:
                print(f"建議減少Arial字體大小約 {int(-height_diff / 2)}像素，或增加Ebrima字體大小約 {int(-height_diff / 2)}像素")
        
        # 建議基準Y位置調整
        y_adjustment = height_diff / 2
        if abs(y_adjustment) > 2:
            print(f"建議調整Arial基準Y位置約 {int(y_adjustment)}像素")
        
        # 建議粗度調整
        if abs(thickness_diff) > 2:
            if thickness_diff > 0:
                print(f"建議增加Arial加粗因子約 {thickness_diff / avg_arial_thickness:.2f}，或減少Ebrima加粗因子約 {thickness_diff / avg_ebrima_thickness:.2f}")
            else:
                print(f"建議減少Arial加粗因子約 {-thickness_diff / avg_arial_thickness:.2f}，或增加Ebrima加粗因子約 {-thickness_diff / avg_ebrima_thickness:.2f}")

def main():
    captcha_dir = r"C:\Users\C14511\Desktop\captcha\CAPTCHA\000"
    
    print(f"開始分析目錄: {captcha_dir}")
    char_stats = batch_analyze_captchas(captcha_dir)
    
    if char_stats:
        visualize_char_stats(char_stats)
    else:
        print("分析失敗，未獲得有效的字符統計資料")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()