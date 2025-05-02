import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
from collections import defaultdict
import pandas as pd

# 需要導入原始的模型類和相關定義
class CaptchaCNN(nn.Module):
    def __init__(self, num_chars, num_classes, image_height, image_width, dropout_rate=0.3):
        super(CaptchaCNN, self).__init__()
        self.dropout_rate = dropout_rate
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        h_out = image_height // 8  
        w_out = image_width // 8
        
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(128 * h_out * w_out, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, num_chars * num_classes)
        )
        
        self.num_chars = num_chars
        self.num_classes = num_classes
                
    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        x = x.view(batch_size, self.num_chars, self.num_classes)
        return x

class EnsembleModel(nn.Module):
    def __init__(self, model1, model2):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        
    def forward(self, x):
        output1 = self.model1(x)
        output2 = self.model2(x)
        # 平均兩個模型的輸出
        return (output1 + output2) / 2

class CaptchaDataset(Dataset):
    def __init__(self, img_dir, char_set, transform=None, captcha_length=6):
        self.img_dir = img_dir
        self.transform = transform
        self.captcha_length = captcha_length
        self.char_set = sorted(char_set)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.char_set)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.char_set)}
        
        # 獲取所有圖片文件
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 讀取圖片
        image = Image.open(img_path).convert('RGB')
        
        # 從文件名中提取標籤
        label = img_name.split('.')[0]
        
        # 將標籤轉換為索引
        label_indices = torch.zeros(self.captcha_length, dtype=torch.long)
        for i, char in enumerate(label[:self.captcha_length]):
            if i < self.captcha_length:
                label_indices[i] = self.char_to_idx.get(char, 0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_indices

def test_ensemble_model(test_dir, model_path, config_path, output_dir='test_results'):
    """測試集成模型的主函數"""
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 載入配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 設定裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")
    
    # 載入檢查點
    checkpoint = torch.load(model_path, map_location=device)
    print(f"成功載入模型檢查點: {model_path}")
    
    # 設定資料轉換
    transform = transforms.Compose([
        transforms.Resize((config['image_height'], config['image_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 建立測試資料集
    test_dataset = CaptchaDataset(
        img_dir=test_dir,
        char_set=config['char_set'],
        transform=transform,
        captcha_length=config['captcha_length']
    )
    
    # 建立資料載入器
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # 建立模型
    model_original = CaptchaCNN(
        num_chars=config['captcha_length'],
        num_classes=len(config['char_set']),
        image_height=config['image_height'],
        image_width=config['image_width']
    ).to(device)
    
    model_denoised = CaptchaCNN(
        num_chars=config['captcha_length'],
        num_classes=len(config['char_set']),
        image_height=config['image_height'],
        image_width=config['image_width']
    ).to(device)
    
    # 載入權重
    model_original.load_state_dict(checkpoint['model_original'])
    model_denoised.load_state_dict(checkpoint['model_denoised'])
    
    # 創建集成模型
    ensemble_model = EnsembleModel(model_original, model_denoised).to(device)
    ensemble_model.eval()
    
    # 進行測試
    correct_chars = 0
    correct_strings = 0
    total_chars = 0
    total_strings = 0
    
    all_predictions = []
    all_labels = []
    all_img_files = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="測試中")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            outputs = ensemble_model(images)
            _, predicted = torch.max(outputs.data, 2)
            
            # 計算字元準確率
            correct_chars += (predicted == labels).sum().item()
            total_chars += labels.size(0) * labels.size(1)
            
            # 計算字串準確率
            correct_mask = (predicted == labels).all(dim=1)
            correct_strings += correct_mask.sum().item()
            total_strings += labels.size(0)
            
            # 保存預測結果
            for i in range(labels.size(0)):
                pred_string = ''.join([test_dataset.char_set[idx] for idx in predicted[i].cpu().numpy()])
                true_string = ''.join([test_dataset.char_set[idx] for idx in labels[i].cpu().numpy()])
                all_predictions.append(pred_string)
                all_labels.append(true_string)
                # 獲取對應的圖片文件名
                img_idx = batch_idx * config['batch_size'] + i
                if img_idx < len(test_dataset.img_files):
                    all_img_files.append(test_dataset.img_files[img_idx])
                else:
                    all_img_files.append(f"unknown_{img_idx}")
            
            progress_bar.set_postfix({
                'char_acc': f"{100 * correct_chars / total_chars:.2f}%",
                'string_acc': f"{100 * correct_strings / total_strings:.2f}%"
            })
    
    # 計算最終準確率
    char_accuracy = 100 * correct_chars / total_chars
    string_accuracy = 100 * correct_strings / total_strings
    
    # 生成測試報告
    generate_test_report(all_labels, all_predictions, all_img_files, test_dataset.char_set, 
                        char_accuracy, string_accuracy, output_dir)
    
    # 視覺化預測結果
    visualize_predictions(ensemble_model, test_loader, test_dataset.char_set, device, output_dir)
    
    return char_accuracy, string_accuracy

def generate_test_report(labels, predictions, img_files, char_set, char_accuracy, string_accuracy, output_dir):
    """生成詳細的測試報告"""
    report_path = os.path.join(output_dir, 'test_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # 測試結果
        f.write("=== 測試結果 ===\n")
        f.write(f"總樣本數: {len(labels)}\n")
        f.write(f"字元準確率: {char_accuracy:.2f}%\n")
        f.write(f"字串準確率: {string_accuracy:.2f}%\n\n")
        
        # 預測結果與真實標籤
        f.write("=== 預測結果與真實標籤 ===\n")
        f.write(f"{'樣本':<20} {'真實值':<10} {'預測值':<10} {'是否正確':<10}\n")
        f.write("-" * 50 + "\n")
        
        for i, (true_label, pred_label, img_file) in enumerate(zip(labels, predictions, img_files)):
            is_correct = "正確" if true_label == pred_label else "錯誤"
            f.write(f"{img_file:<20} {true_label:<10} {pred_label:<10} {is_correct:<10}\n")
            
            # 只顯示前50個樣本，避免文件過大
            if i >= 49:
                f.write("...\n")
                f.write(f"(共 {len(labels)} 個樣本)\n")
                break
        
        # 錯誤分析報告
        f.write("\n=== 錯誤分析報告 ===\n")
        total_errors = 0
        position_errors = defaultdict(int)
        char_confusion = defaultdict(lambda: defaultdict(int))
        char_errors = defaultdict(int)
        
        # 統計錯誤
        for true_label, pred_label in zip(labels, predictions):
            if true_label != pred_label:
                total_errors += 1
                for i, (true_char, pred_char) in enumerate(zip(true_label, pred_label)):
                    if true_char != pred_char:
                        position_errors[i] += 1
                        char_confusion[true_char][pred_char] += 1
                        char_errors[true_char] += 1
        
        # 總體錯誤率
        error_rate = 100 * total_errors / len(labels)
        f.write(f"總體錯誤率: {error_rate:.2f}% ({total_errors}/{len(labels)})\n\n")
        
        # 位置錯誤分析
        f.write("=== 位置錯誤分析 ===\n")
        f.write(f"{'位置':<8} {'錯誤次數':<10} {'錯誤率':<10}\n")
        f.write("-" * 30 + "\n")
        
        for i in range(len(labels[0])):  # 假設所有標籤長度相同
            pos_error_count = position_errors[i]
            pos_error_rate = 100 * pos_error_count / len(labels)
            f.write(f"{i+1:<8} {pos_error_count:<10} {pos_error_rate:.2f}%\n")
        
        # 各字元錯誤統計
        f.write("\n=== 各字元錯誤統計 ===\n")
        f.write(f"{'真實字元':<10} {'錯誤次數':<10} {'最常被誤認為':<15}\n")
        f.write("-" * 40 + "\n")
        
        for true_char in sorted(char_errors.keys(), key=lambda x: char_errors[x], reverse=True):
            error_count = char_errors[true_char]
            if error_count > 0:
                # 找出最常被誤認為的字元
                most_confused = max(char_confusion[true_char].items(), key=lambda x: x[1])
                f.write(f"{true_char:<10} {error_count:<10} {most_confused[0]} ({most_confused[1]}次)\n")
        
        # 詳細的字元混淆矩陣
        f.write("\n=== 字元混淆矩陣 ===\n")
        for true_char in sorted(char_confusion.keys()):
            f.write(f"\n真實字元 '{true_char}' 被誤認為:\n")
            for pred_char, count in sorted(char_confusion[true_char].items(), key=lambda x: x[1], reverse=True):
                f.write(f"  '{pred_char}': {count} 次\n")
    
    # 同時保存為 CSV 格式
    results_df = pd.DataFrame({
        '樣本': img_files,
        '真實值': labels,
        '預測值': predictions,
        '是否正確': [pred == true for pred, true in zip(predictions, labels)]
    })
    results_df.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False, encoding='utf-8-sig')
    
    print(f"測試報告已保存至: {report_path}")
    print(f"詳細結果已保存至: {os.path.join(output_dir, 'test_results.csv')}")

def visualize_predictions(model, test_loader, char_set, device, output_dir, num_samples=20):
    """視覺化預測結果"""
    vis_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    
    model.eval()
    samples_seen = 0
    
    # 分別保存正確和錯誤的樣本
    correct_dir = os.path.join(vis_dir, 'correct')
    error_dir = os.path.join(vis_dir, 'errors')
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)
    
    error_count = 0
    correct_count = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            if samples_seen >= num_samples:
                break
            
            images_device = images.to(device)
            outputs = model(images_device)
            _, predictions = torch.max(outputs.data, 2)
            
            for i in range(min(images.size(0), num_samples - samples_seen)):
                # 反正規化圖片
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                
                # 獲取預測和真實標籤
                pred_text = ''.join([char_set[idx] for idx in predictions[i].cpu().numpy()])
                true_text = ''.join([char_set[idx] for idx in labels[i].cpu().numpy()])
                
                # 繪製圖片
                plt.figure(figsize=(10, 4))
                plt.imshow(img)
                plt.title(f"真實值: {true_text} | 預測值: {pred_text}")
                plt.axis('off')
                plt.tight_layout()
                
                # 根據預測結果分類保存
                if pred_text == true_text:
                    save_path = os.path.join(correct_dir, f"correct_{correct_count + 1}_{true_text}.png")
                    correct_count += 1
                else:
                    save_path = os.path.join(error_dir, f"error_{error_count + 1}_{true_text}_pred_{pred_text}.png")
                    error_count += 1
                
                plt.savefig(save_path)
                plt.close()
                
            samples_seen += images.size(0)
    
    print(f"視覺化結果已保存至: {vis_dir}")

def main():
    # 設定測試參數
    test_dir = r"C:\Users\C14511\Desktop\captcha\CAPTCHA\00000"  # 測試資料夾路徑
    model_path = r"models\best_ensemble_model.pth"  # 訓練好的模型路徑
    config_path = r"config.json"  # 配置文件路徑
    output_dir = r"test_results"  # 輸出目錄
    
    # 執行測試
    char_accuracy, string_accuracy = test_ensemble_model(test_dir, model_path, config_path, output_dir)
    
    print(f"\n測試完成！")
    print(f"字元準確率: {char_accuracy:.2f}%")
    print(f"字串準確率: {string_accuracy:.2f}%")
    print(f"結果已保存至: {output_dir}")

if __name__ == "__main__":
    main()