# 載入需要的函式庫
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import requests
import shutil

# 按字體區分字符集
arial_characters = ['3', '4', '6', '8', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'm', 'p', 'r', 'w', 'x', 'y']
ebrima_characters = ['2', '5', '7', 'a', 'k']

print("Arial字符集:", ''.join(arial_characters))
print("Ebrima字符集:", ''.join(ebrima_characters))

# 下載字體的函數
def find_font(font_name, font_dirs=None):
    """尋找指定的字體文件"""
    if font_dirs is None:
        font_dirs = [
            "fonts",  # 本地專案目錄
            os.path.join(os.getcwd(), "fonts"),  # 絕對路徑
            os.path.join(os.path.expanduser("~"), "fonts"),  # 用戶目錄
            "C:/Windows/Fonts"  # Windows系統字體
        ]
    
    # 確保字體目錄存在
    os.makedirs("fonts", exist_ok=True)
    
    # 搜索指定的字體
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            font_path = os.path.join(font_dir, font_name)
            if os.path.exists(font_path):
                print(f"找到字體: {font_path}")
                return font_path
    
    # 搜索不區分大小寫的字體名稱
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for file in os.listdir(font_dir):
                if file.lower() == font_name.lower() or file.lower().startswith(font_name.lower().split('.')[0]):
                    font_path = os.path.join(font_dir, file)
                    print(f"找到字體: {font_path}")
                    return font_path
    
    return None

# 定義CAPTCHA生成器類
class GrayCaptchaGenerator:
    def __init__(self, width=225, height=69, height_compression=0.5):
        self.final_width = width
        self.final_height = height
        # 創建一個更高的初始圖像
        self.initial_height = int(height * 1.1)  # 初始高度是目標高度的1.1倍
        self.initial_width = width
        self.height_compression = height_compression
    
        # 尋找Arial和Ebrima字體
        self.arial_font_path = find_font("arial.ttf")
        self.ebrima_font_path = find_font("ebrima.ttf")
        
        if not self.arial_font_path:
            print("警告: 未找到Arial字體")
        else:
            print(f"使用Arial字體: {self.arial_font_path}")
            
        if not self.ebrima_font_path:
            print("警告: 未找到Ebrima字體")
        else:
            print(f"使用Ebrima字體: {self.ebrima_font_path}")
        
        # 設定字體大小
        self.arial_font_size = 58    # Arial字體比較細，稍微大一點
        self.ebrima_font_size = 48   # Ebrima字體比較粗，稍微小一點
        
        # 為兩種字體設定垂直偏移量，使其在同一水平線上
        self.arial_y_offset = 5      # Arial字體的垂直偏移
        self.ebrima_y_offset = 3     # Ebrima字體的垂直偏移，根據需要調整
    
    def generate_captcha(self, text=None, length=6):
        """生成CAPTCHA圖像"""
        # 如果未提供文字，隨機生成
        if text is None:
            # 首先生成純文字字符串
            all_characters = arial_characters + ebrima_characters
            captcha_text = ''.join(random.choice(all_characters) for _ in range(length))
            
            # 為每個字符選擇合適的字體
            font_choices = []
            for char in captcha_text:
                if char in arial_characters:
                    font_choices.append('arial')
                elif char in ebrima_characters:
                    font_choices.append('ebrima')
                else:
                    # 如果字符不在任何集合中，預設使用Arial
                    font_choices.append('arial')
        else:
            # 如果提供了文字，解析字體選擇
            # 這裡需要根據您的需求定義如何處理提供的文字
            captcha_text = text
            font_choices = ['arial'] * len(text)  # 預設全部使用Arial
        
        # 使用純文字進行顯示
        display_text = captcha_text
        
        # 創建中灰色背景
        background_color = (210, 210, 210)
        image = Image.new('RGB', (self.initial_width, self.initial_height), background_color)

        # 計算字符寬度和位置
        char_widths = []
        fonts = []
        
        for i, char in enumerate(display_text):
            if i < len(font_choices):
                if font_choices[i] == 'arial' and self.arial_font_path:
                    font = ImageFont.truetype(self.arial_font_path, self.arial_font_size)
                elif font_choices[i] == 'ebrima' and self.ebrima_font_path:
                    font = ImageFont.truetype(self.ebrima_font_path, self.ebrima_font_size)
                else:
                    # 如果指定的字體不可用，使用任何可用的字體
                    if self.arial_font_path:
                        font = ImageFont.truetype(self.arial_font_path, self.arial_font_size)
                    elif self.ebrima_font_path:
                        font = ImageFont.truetype(self.ebrima_font_path, self.ebrima_font_size)
                    else:
                        raise ValueError("沒有可用的字體")
                
                fonts.append(font)
                bbox = font.getbbox(char)
                char_width = bbox[2] - bbox[0]
                char_widths.append(char_width)
        
        # 創建中灰色背景
        background_color = (210, 210, 210)
        image = Image.new('RGB', (self.initial_width, self.initial_height), background_color)
        
        # 設置固定間距
        fixed_spacing = 0
        
        # 計算文本總寬度
        total_width = sum(char_widths) + fixed_spacing * (len(display_text) - 1)
        
        # 計算起始位置，確保文本居中
        start_x = (self.initial_width - total_width) // 2
        
        # 添加一些整體的水平隨機位移
        horizontal_shift = random.randint(-10, 10)
        start_x += horizontal_shift

        # 設定基準Y位置 - 所有字符放在同一水平線上
        # 將基準線調整到圖片中間偏上位置
        baseline_y = int(self.initial_height * 0)  # 調整這個值來改變基準線位置
        
        # 在放置字元時記錄位置
        char_positions = []
        x = start_x  # 起始X位置

        # 為每個字符創建單獨的圖層
        char_images = []
        
        for i, char in enumerate(display_text):
            font = fonts[i]
            char_width = char_widths[i]
            
            # 獲取字符的高度
            bbox = font.getbbox(char)
            char_height = bbox[3] - bbox[1]

            # 根據字體類型應用不同的垂直偏移
            if font_choices[i] == 'arial':
                y_offset = self.arial_y_offset
            else:  # ebrima或其他字體
                y_offset = self.ebrima_y_offset
            
            # 計算字符Y位置，考慮偏移量
            char_y = baseline_y + y_offset
            
            # 記錄字符位置
            char_positions.append((x, char_y, x + char_width, char_y + char_height))

            # 為單個字符創建透明圖層
            char_img = Image.new('RGBA', (self.initial_width, self.initial_height), (0, 0, 0, 0))
            char_draw = ImageDraw.Draw(char_img)

            # 字體加粗調整（Arial需要更多加粗，Ebrima較少）
            if font_choices[i] == 'arial':
                bold_factor = 0.8
            else:
                bold_factor = 1.0
                
            offsets = []
            for dx in range(-int(bold_factor), int(bold_factor) + 1):
                for dy in range(-int(bold_factor), int(bold_factor) + 1):
                    if dx*dx + dy*dy <= bold_factor*bold_factor:
                        offsets.append((dx, dy))

            # 繪製所有偏移版本
            for offset_x, offset_y in offsets:
                char_draw.text((x + offset_x, char_y + offset_y), char, font=font, fill=(30, 30, 30, 255))
            
            # 再次在原始位置繪製，確保字符核心最黑
            char_draw.text((x, char_y), char, font=font, fill=(0, 0, 0, 255))

            # 應用圓弧化處理
            char_img_blurred = char_img.filter(ImageFilter.GaussianBlur(radius=1.0))

            # 調整透明度閾值
            char_img_processed = Image.new('RGBA', char_img.size, (0, 0, 0, 0))
            char_data = list(char_img_blurred.getdata())
            processed_data = []
            for pixel in char_data:
                if pixel[3] > 20:
                    processed_data.append((pixel[0], pixel[1], pixel[2], min(255, int(pixel[3] * 1.3))))
                else:
                    processed_data.append((0, 0, 0, 0))
            char_img_processed.putdata(processed_data)

            # 保存處理後的字符圖像
            char_images.append((char_img_processed, x))
            
            # 更新下一個字符的X位置，加上當前字符寬度和固定間距
            x += char_width + fixed_spacing

        # 將所有處理過的字符合併到最終圖像
        final_image = Image.new('RGB', (self.initial_width, self.initial_height), background_color)
        for char_img, x_pos in char_images:
            final_image.paste((0, 0, 0), (0, 0), char_img)
    
        # 在字符合併後加入干擾線
        dark_gray = (50, 50, 50)  # 更深的灰色，接近黑色
        line_width = 10
        
        # 選擇一個隨機字元進行穿過
        selected_char = random.choice(char_positions)
        char_x_center = (selected_char[0] + selected_char[2]) // 2
        char_y_center = (selected_char[1] + selected_char[3]) // 2

        # 創建控制點，確保穿過選擇的字元
        points = []

        # 起始點可以在圖像的任何左側位置
        start_x = random.randint(0, self.initial_width // 4)
        start_y = random.randint(0, self.initial_height)
        points.append((start_x, start_y))

        # 添加一個穿過字元的點
        char_point_x = char_x_center + random.randint(-5, 5)
        char_point_y = char_y_center + random.randint(-5, 5)
        points.append((char_point_x, char_point_y))

        # 結束點可以在圖像的任何右側位置
        end_x = random.randint(3 * self.initial_width // 4, self.initial_width)
        end_y = random.randint(0, self.initial_height)
        points.append((end_x, end_y))

        # 創建一個透明圖層來繪製干擾線
        line_img = Image.new('RGBA', (self.initial_width, self.initial_height), (0, 0, 0, 0))
        line_draw = ImageDraw.Draw(line_img)

        # 在透明圖層上繪製線條
        for i in range(len(points) - 1):
            line_draw.line([points[i], points[i+1]], fill=dark_gray, width=line_width)

        # 對干擾線圖層應用圓弧化處理
        line_img_blurred = line_img.filter(ImageFilter.GaussianBlur(radius=1.2))

        # 調整透明度閾值
        line_img_processed = Image.new('RGBA', line_img.size, (0, 0, 0, 0))
        line_data = list(line_img_blurred.getdata())
        processed_line_data = []

        for pixel in line_data:
            if pixel[3] > 30:  # 如果有足夠的透明度
                # 保持線條的顏色，但根據原始透明度調整新的透明度
                processed_line_data.append((dark_gray[0], dark_gray[1], dark_gray[2], min(255, int(pixel[3] * 1.2))))
            else:
                processed_line_data.append((0, 0, 0, 0))

        line_img_processed.putdata(processed_line_data)

        # 將處理過的干擾線合併到最終圖像
        final_image.paste((dark_gray[0], dark_gray[1], dark_gray[2]), (0, 0), line_img_processed)

        # 繼續應用最終的模糊效果
        # 使用較小的模糊半徑以保持字體清晰
        final_image = final_image.filter(ImageFilter.GaussianBlur(radius=0.8))  # 降低最終模糊半徑
        
        # 最後將圖像縮放到目標大小，這將實現垂直壓縮效果
        final_image = final_image.resize((self.final_width, self.final_height), Image.LANCZOS)

        return final_image, captcha_text

    def batch_generate(self, count, output_dir="captchas"):
        """批量生成CAPTCHA圖像並保存"""
        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 創建CSV文件來儲存標籤
        csv_path = os.path.join(output_dir, "labels.csv")
        with open(csv_path, "w") as f:
            f.write("filename,text\n")  # 寫入CSV標頭
            
            # 批量生成圖像
            start_time = time.time()
            for i in range(count):
                # 顯示進度
                if (i+1) % 100 == 0 or i == 0 or i == count-1:
                    percent = ((i+1) / count) * 100
                    elapsed = time.time() - start_time
                    images_per_sec = (i+1) / elapsed if elapsed > 0 else 0
                    est_remaining = (count - (i+1)) / images_per_sec if images_per_sec > 0 else 0
                    print(f"進度: {i+1}/{count} ({percent:.1f}%) - "
                          f"速率: {images_per_sec:.1f}張/秒 - "
                          f"預計剩餘時間: {est_remaining:.1f}秒")
                
                # 生成圖像
                img, text = self.generate_captcha()
                
                # 創建新的檔名格式: 直接使用答案.png
                filename = f"{text}.png"
                filepath = os.path.join(output_dir, filename)
                
                # 保存圖像
                img.save(filepath)
                
                # 寫入CSV標籤
                f.write(f"{filename},{text}\n")
        
        print(f"\n批量生成完成！共生成了{count}張CAPTCHA圖像。")
        print(f"圖像和標籤已保存到 {os.path.abspath(output_dir)} 目錄。")
        return os.path.abspath(output_dir)

# 主程式
if __name__ == "__main__":
    try:
        # 直接指定 Ebrima 字體的系統路徑
        ebrima_font_path = r"C:\Windows\Fonts\ebrima.ttf"
        
        # 檢查字體是否存在
        if not os.path.exists(ebrima_font_path):
            print(f"在系統中找不到 Ebrima 字體: {ebrima_font_path}")
            print("嘗試使用備用字體...")
            # 嘗試在本地 fonts 目錄查找
            local_ebrima = os.path.join("fonts", "ebrima.ttf")
            if os.path.exists(local_ebrima):
                ebrima_font_path = local_ebrima
            else:
                print("請將 ebrima.ttf 字體文件放在 fonts 目錄下")
                # 嘗試使用任何可用的系統字體
                if os.path.exists("C:/Windows/Fonts"):
                    system_fonts = [f for f in os.listdir("C:/Windows/Fonts") if f.lower().endswith('.ttf')]
                    if system_fonts:
                        ebrima_font_path = os.path.join("C:/Windows/Fonts", system_fonts[0])
                        print(f"使用系統字體作為備用: {ebrima_font_path}")
                       
        # 創建CAPTCHA生成器實例
        captcha_generator = GrayCaptchaGenerator()

        # 批量生成CAPTCHA圖像
        num_images = 500  # 要生成的圖像數量
        output_dir = r"C:\Users\C14511\Desktop\captcha\CAPTCHA\00000"  # 輸出目錄
        
        # 生成樣例預覽
        print("預覽生成的CAPTCHA樣例:")
        plt.figure(figsize=(15, 10))
        
        for i in range(4):
            img, text = captcha_generator.generate_captcha()
            
            plt.subplot(2, 2, i+1)
            plt.imshow(img)
            plt.title(text)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 詢問用戶是否繼續批量生成
        user_input = input(f"確定要生成 {num_images} 張CAPTCHA圖像嗎？(y/n): ")
        
        if user_input.lower() == 'y':
            # 開始批量生成
            save_dir = captcha_generator.batch_generate(num_images, output_dir)
            print(f"所有圖像已保存到: {save_dir}")
        else:
            print("已取消批量生成。")
        
    except Exception as e:
        print(f"發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()