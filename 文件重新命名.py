#文件重新命名
import os


def rename_files(directory, prefix):
    # 獲取目錄中的所有文件
    files = [f for f in os.listdir(directory) if f.lower().endswith('.png')]
   
    # 遍歷文件並重新命名
    for i, filename in enumerate(files, start=1):
        new_name = f"{prefix}{i}.png"
        src = os.path.join(directory, filename)
        dst = os.path.join(directory, new_name)
        os.rename(src, dst)
        print(f"Renamed '{filename}' to '{new_name}'")


# 指定目錄
directory = r"C:\Users\C14511\Desktop\captcha\CAPTCHA\abcdef\y"


# 設定自訂命名前綴
prefix = "y_1000 "


# 執行重命名
rename_files(directory, prefix)