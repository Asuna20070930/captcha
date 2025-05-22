import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import json
from torchvision.models import resnet18
import sys
from tqdm import tqdm
import logging
import psutil
import random

class TrainingLogger:
    def __init__(self, log_dir='logs'):
        os.makedirs(log_dir, exist_ok=True)
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f"training_log_{current_time}.log")
        self.summary_file = os.path.join(log_dir, f"training_summary_{current_time}.txt")

        # Set detailed log
        logging.basicConfig(filename=self.log_file,
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
        self.logger = logging.getLogger(__name__)

        # Set concise summary log
        self.summary_fh = open(self.summary_file, 'w', encoding='utf-8')
        self.summary_fh.write(f"訓練開始時間: {current_time}\n\n")

    def log(self, message):
        self.logger.info(message)
        print(message)

    def summary(self, message):
        timestamp = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        summary_message = f"{timestamp}: {message}"
        self.summary_fh.write(f"{summary_message}\n")
        self.summary_fh.flush()  # Ensure immediate write to file
        self.log(f"[摘要] {message}")

    def close(self):
        if hasattr(self, 'summary_fh'):
            self.summary_fh.close()

def analyze_errors(val_loader, predicted_labels, true_labels, logger, output_dir='error_analysis'):
    os.makedirs(output_dir, exist_ok=True)
    error_count = 0
    position_errors = {i: 0 for i in range(len(true_labels[0]))}
    char_confusion = {}
    
    for i in range(len(true_labels)):
        pred = predicted_labels[i]
        true = true_labels[i]
        if pred != true:
            error_count += 1
            for pos, (p, t) in enumerate(zip(pred, true)):
                if p != t:
                    position_errors[pos] += 1
                    if t not in char_confusion:
                        char_confusion[t] = {}
                    if p not in char_confusion[t]:
                        char_confusion[t][p] = 0
                    char_confusion[t][p] += 1

    logger.log(f"錯誤樣本總數: {error_count} / {len(true_labels)} ({error_count/len(true_labels)*100:.2f}%)")
    logger.log("最常見的字元混淆 (前15個):")
    confusion_list = []
    for true_char in char_confusion:
        for pred_char, count in char_confusion[true_char].items():
            confusion_list.append((true_char, pred_char, count))
    confusion_list.sort(key=lambda x: x[2], reverse=True)
    for true_char, pred_char, count in confusion_list[:15]:
        logger.log(f"  字元 '{true_char}' 錯誤預測為 '{pred_char}': {count} 次")
    
    with open(os.path.join(output_dir, 'char_confusion.csv'), 'w', encoding='utf-8') as f:
        f.write("TrueChar,PredictedChar,Count\n")
        for true_char, pred_char, count in confusion_list:
            f.write(f"{true_char},{pred_char},{count}\n")
    
    logger.log("分析每個位置的錯誤...")
    for pos in range(len(true_labels[0])):
        if position_errors[pos] > 0:
            logger.log(f"位置 {pos+1}:")
            pos_confusion = {}
            for i in range(len(true_labels)):
                if predicted_labels[i][pos] != true_labels[i][pos]:
                    t = true_labels[i][pos]
                    p = predicted_labels[i][pos]
                    if t not in pos_confusion:
                        pos_confusion[t] = {}
                    if p not in pos_confusion[t]:
                        pos_confusion[t][p] = 0
                    pos_confusion[t][p] += 1
            with open(os.path.join(output_dir, f'position_{pos+1}_confusion.csv'), 'w', encoding='utf-8') as f:
                f.write("TrueChar,PredictedChar,Count\n")
                for true_char in pos_confusion:
                    for pred_char, count in pos_confusion[true_char].items():
                        f.write(f"{true_char},{pred_char},{count}\n")
                        logger.log(f"  字元 '{true_char}' 錯誤預測為 '{pred_char}': {count} 次")
    
    total_errors = sum(position_errors.values())
    logger.log("位置錯誤分佈:")
    for pos, count in position_errors.items():
        if count > 0:
            percentage = (count / total_errors) * 100
            logger.log(f"  位置 {pos+1}: {count} 錯誤 ({percentage:.1f}%)")
    
    plt.figure(figsize=(10, 6))
    positions = list(position_errors.keys())
    counts = [position_errors[pos] for pos in positions]
    plt.bar([str(p+1) for p in positions], counts)
    plt.xlabel('字元位置')
    plt.ylabel('錯誤數量')
    plt.title('各位置錯誤分佈')
    plt.savefig(os.path.join(output_dir, 'position_errors.png'))
    
    return error_count, position_errors, char_confusion

class CaptchaDataset(Dataset):
    def __init__(self, img_dir, char_set, transform=None, captcha_length=6):
        self.img_dir = img_dir
        self.transform = transform
        self.char_set = char_set
        self.char_to_idx = {char: idx for idx, char in enumerate(char_set)}
        self.captcha_length = captcha_length
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = img_name.split('_')[0]
        label_indices = torch.tensor([self.char_to_idx[c] for c in label[:self.captcha_length]])
        if self.transform:
            image = self.transform(image)
        return image, label_indices

class CustomConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        # 假設所有數據集都有相同的 char_set
        if hasattr(datasets[0], 'char_set'):
            self.char_set = datasets[0].char_set
    
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
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        x = x.view(batch_size, self.num_chars, self.num_classes)
        return x

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input.view(-1, input.size(-1)), target.view(-1), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class MixupAugmentation:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, images, labels):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        mixed_images = lam * images + (1 - lam) * images[index]
        return mixed_images, labels, labels[index], lam

def train(model, train_loader, criterion, optimizer, scheduler, device, epoch, total_epochs, logger, use_mixup=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} 訓練中")
    mixup = MixupAugmentation(alpha=0.2) if use_mixup else None
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        if use_mixup:
            mixed_images, labels_a, labels_b, lam = mixup(images, labels)
            outputs = model(mixed_images)
            loss_a = criterion(outputs.view(-1, outputs.size(-1)), labels_a.view(-1))
            loss_b = criterion(outputs.view(-1, outputs.size(-1)), labels_b.view(-1))
            loss = lam * loss_a + (1 - lam) * loss_b
        else:
            outputs = model(images)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 2)
        total += labels.size(0) * labels.size(1)
        correct += (predicted == labels).sum().item()
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100 * correct / total:.2f}%"
        })

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    scheduler.step()

    logger.log(f"訓練週期 {epoch}/{total_epochs}: 訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.2f}%")
    return train_loss, train_acc

def validate(model, val_loader, criterion, device, epoch, total_epochs, logger):
    model.eval()
    running_loss = 0.0
    correct_chars = 0
    correct_strings = 0
    total_chars = 0
    total_strings = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{total_epochs} 驗證中")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 2)
            correct_chars += (predicted == labels).sum().item()
            total_chars += labels.size(0) * labels.size(1)
            correct_mask = (predicted == labels).all(dim=1)
            correct_strings += correct_mask.sum().item()
            total_strings += labels.size(0)
            for i in range(labels.size(0)):
                pred_string = ''.join([val_loader.dataset.char_set[idx] for idx in predicted[i].cpu().numpy()])
                true_string = ''.join([val_loader.dataset.char_set[idx] for idx in labels[i].cpu().numpy()])
                all_predictions.append(pred_string)
                all_labels.append(true_string)
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct_strings / total_strings:.2f}%"
            })
    
    test_loss = running_loss / len(val_loader)
    char_accuracy = 100 * correct_chars / total_chars
    string_accuracy = 100 * correct_strings / total_strings
    
    logger.log(f"驗證週期 {epoch}/{total_epochs}:")
    logger.log(f"  測試損失: {test_loss:.4f}")
    logger.log(f"  字元準確率: {char_accuracy:.2f}%")
    logger.log(f"  字串準確率: {string_accuracy:.2f}%")
    
    # 簡潔摘要日誌
    logger.summary(f"驗證損失: {test_loss:.4f}, 字元準確率: {char_accuracy:.2f}%, 字串準確率: {string_accuracy:.2f}%")
    
    return test_loss, string_accuracy, char_accuracy, all_predictions, all_labels

def visualize_predictions(model, test_loader, char_set, device, num_samples=10, output_dir="prediction_viz"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    samples_seen = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if samples_seen >= num_samples:
                break
            batch_size = images.size(0)
            actual_samples = min(batch_size, num_samples - samples_seen)
            images_device = images[:actual_samples].to(device)
            outputs = model(images_device)
            _, predictions = torch.max(outputs.data, 2)
            for i in range(actual_samples):
                img = images[i].permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                pred_indices = predictions[i].cpu().numpy()
                true_indices = labels[i].cpu().numpy()
                pred_text = ''.join([char_set[idx] for idx in pred_indices])
                true_text = ''.join([char_set[idx] for idx in true_indices])
                plt.figure(figsize=(10, 4))
                plt.imshow(img)
                plt.title(f"真實值: {true_text} | 預測值: {pred_text}")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"sample_{samples_seen + i + 1}.png"))
                plt.close()
            samples_seen += actual_samples

def find_latest_checkpoint(checkpoint_dir, logger):
    """尋找最新儲存的檢查點檔案"""
    if not os.path.exists(checkpoint_dir):
        return None
        
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")]
    if not checkpoint_files:
        return None

    # 使用數字排序，確保找到最新的檢查點
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
    logger.log(f"找到最新的檢查點: {latest_checkpoint}")
    return latest_checkpoint

def load_checkpoint(model, optimizer, scheduler, checkpoint_dir, logger):
    try:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir, logger)
        if not checkpoint_path:
            logger.log(f"檢查點檔案未找到在 {checkpoint_dir}, 從頭開始訓練。")
            return 0, 0

        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 只在optimizer存在時才載入
        if 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                logger.log("無法載入optimizer狀態，使用當前初始化的optimizer")
        
        # 只在scheduler存在時才載入
        if 'scheduler' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
            except:
                logger.log("無法載入scheduler狀態，使用當前初始化的scheduler")
        
        start_epoch = checkpoint['epoch'] + 1
        best_string_accuracy = checkpoint.get('string_accuracy', 0)
        logger.log(f"成功從 {checkpoint_path} 載入檢查點，從 epoch {start_epoch} 繼續訓練")
        return start_epoch, best_string_accuracy
    except Exception as e:
        logger.log(f"載入檢查點時發生錯誤: {str(e)}, 從頭開始訓練。")
        import traceback
        logger.log(traceback.format_exc())
        return 0, 0

def save_checkpoint(model, optimizer, scheduler, epoch, string_accuracy, char_accuracy, logger, best_model=False, config=None, train_dataset=None):
    filename = "best_model.pth" if best_model else f"checkpoint_epoch_{epoch}.pth"
    checkpoint_path = os.path.join("models", filename)
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'string_accuracy': string_accuracy,
            'char_accuracy': char_accuracy,
            'config': config,
            'char_set': train_dataset.char_set if train_dataset else None
        }, checkpoint_path)
        logger.log(f"成功儲存檢查點至 {checkpoint_path}")
    except Exception as e:
        logger.log(f"儲存檢查點時發生錯誤: {str(e)}")

def main():
    config = {
        'train_dir_original': r"C:\Users\C14511\Desktop\captcha\CAPTCHA\training",      # 原始影像訓練資料集目錄
        'train_dir_denoised': r"C:\Users\C14511\Desktop\captcha\CAPTCHA\training01",    # 原始去雜訊影像訓練資料集目錄
        'train_dir_custom_original': r"C:\Users\C14511\Desktop\captcha\CAPTCHA\000",    # 自創原始影像訓練資料集目錄
        'train_dir_custom_denoised': r"C:\Users\C14511\Desktop\captcha\CAPTCHA\0000",   # 自創去雜訊影像訓練資料集目錄
        'test_dir': r"C:\Users\C14511\Desktop\captcha\CAPTCHA\training",                # 測試資料集目錄
        'image_width': 225,                                                             # 輸入圖片寬度
        'image_height': 69,                                                             # 輸入圖片高度
        'batch_size': 32,                                                               # 批次大小
        'num_epochs': 600,                                                              # 最大訓練週期
        'learning_rate': 0.001,                                                         # 學習率
        'weight_decay': 0.01,                                                           # 權重衰減
        'captcha_length': 6,                                                            # CAPTCHA 字元長度
        'use_mixup': True,                                                              # 是否使用 Mixup 資料增強
        'use_focalloss': True,                                                          # 是否使用 Focal Loss
        'gamma': 2.0,                                                                   # Focal Loss 的 gamma 參數
        'use_augmentation': True,                                                       # 是否使用額外資料增強
        'early_stopping_patience': 50,                                                  # 提前停止的耐心值
        'checkpoint_interval': 10,                                                      # 定期儲存檢查點的間隔
        'char_set': ['2', '3', '4', '5', '6', '7', '8', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'k', 'm', 'p', 'r', 'w', 'x', 'y'] # 使用的字元集
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    # 設定基本轉換
    transform = transforms.Compose([
        transforms.Resize((config['image_height'], config['image_width'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 設定訓練用的資料增強
    if config['use_augmentation']:
        train_transform = transforms.Compose([
            transforms.Resize((config['image_height'], config['image_width'])),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transform

    logger = TrainingLogger(log_dir='logs')
    logger.log("開始訓練...")
    logger.summary(f"CAPTCHA辨識模型訓練開始")
    logger.summary(f"模型設定: 字元長度={config['captcha_length']}, 字元集大小={len(config['char_set'])}")

    try:
        # 載入四個不同的訓練資料集
        train_datasets = []
        dataset_names = []
        
        # 原始影像訓練資料集
        if os.path.exists(config['train_dir_original']):
            train_dataset_original = CaptchaDataset(
                img_dir=config['train_dir_original'],
                char_set=config['char_set'],
                transform=train_transform,
                captcha_length=config['captcha_length']
            )
            train_datasets.append(train_dataset_original)
            dataset_names.append("原始影像")
            logger.log(f"原始影像訓練集大小: {len(train_dataset_original)}")
        else:
            logger.log(f"警告: 原始影像訓練資料集目錄不存在: {config['train_dir_original']}")
        
        # 原始去雜訊影像訓練資料集
        if os.path.exists(config['train_dir_denoised']):
            train_dataset_denoised = CaptchaDataset(
                img_dir=config['train_dir_denoised'],
                char_set=config['char_set'],
                transform=train_transform,
                captcha_length=config['captcha_length']
            )
            train_datasets.append(train_dataset_denoised)
            dataset_names.append("原始去雜訊影像")
            logger.log(f"原始去雜訊影像訓練集大小: {len(train_dataset_denoised)}")
        else:
            logger.log(f"警告: 原始去雜訊影像訓練資料集目錄不存在: {config['train_dir_denoised']}")
        
        # 自創原始影像訓練資料集
        if os.path.exists(config['train_dir_custom_original']):
            train_dataset_custom_original = CaptchaDataset(
                img_dir=config['train_dir_custom_original'],
                char_set=config['char_set'],
                transform=train_transform,
                captcha_length=config['captcha_length']
            )
            train_datasets.append(train_dataset_custom_original)
            dataset_names.append("自創原始影像")
            logger.log(f"自創原始影像訓練集大小: {len(train_dataset_custom_original)}")
        else:
            logger.log(f"警告: 自創原始影像訓練資料集目錄不存在: {config['train_dir_custom_original']}")
        
        # 自創去雜訊影像訓練資料集
        if os.path.exists(config['train_dir_custom_denoised']):
            train_dataset_custom_denoised = CaptchaDataset(
                img_dir=config['train_dir_custom_denoised'],
                char_set=config['char_set'],
                transform=train_transform,
                captcha_length=config['captcha_length']
            )
            train_datasets.append(train_dataset_custom_denoised)
            dataset_names.append("自創去雜訊影像")
            logger.log(f"自創去雜訊影像訓練集大小: {len(train_dataset_custom_denoised)}")
        else:
            logger.log(f"警告: 自創去雜訊影像訓練資料集目錄不存在: {config['train_dir_custom_denoised']}")
        
        # 檢查是否有有效的訓練資料集
        if not train_datasets:
            logger.log("錯誤: 沒有找到任何有效的訓練資料集!")
            logger.summary("訓練失敗: 沒有找到任何有效的訓練資料集!")
            return
        
        # 合併所有訓練資料集
        train_dataset = CustomConcatDataset(train_datasets)
        
        # 載入測試資料集
        test_dataset = CaptchaDataset(
            img_dir=config['test_dir'],
            char_set=config['char_set'],
            transform=transform,
            captcha_length=config['captcha_length']
        )
        
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            logger.log("錯誤: 找不到有效的訓練或測試資料!")
            logger.summary("訓練失敗: 找不到有效的訓練或測試資料!")
            return

        # 記錄資料集資訊
        total_train_size = len(train_dataset)
        logger.log(f"載入的資料集: {', '.join(dataset_names)}")
        logger.log(f"合併後訓練集總大小: {total_train_size}")
        logger.log(f"測試集大小: {len(test_dataset)}")
        logger.summary(f"資料集: 訓練={total_train_size}個樣本, 測試={len(test_dataset)}個樣本")
        logger.log(f"字元集: {config['char_set']}")
        logger.log(f"字元集大小: {len(config['char_set'])}")

        num_workers = 0 if sys.platform.startswith('win') else 4
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.log(f"使用裝置: {device}")
        logger.summary(f"硬體: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        # 創建單一模型（針對原始影像辨識）
        model = CaptchaCNN(
            num_chars=config['captcha_length'], 
            num_classes=len(config['char_set']),
            image_height=config['image_height'],
            image_width=config['image_width']
        ).to(device)
        
        logger.log("創建CAPTCHA辨識模型（針對原始影像）")
        logger.summary("使用單一模型架構，針對原始影像辨識")

        # 設定損失函數和優化器
        criterion = FocalLoss(gamma=config['gamma']) if config['use_focalloss'] else nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'],
            epochs=config['num_epochs'],
            steps_per_epoch=len(train_loader)
        )

        # 檢查檢查點
        start_epoch = 0
        best_string_accuracy = 0
        checkpoint_files = [f for f in os.listdir("models") if f.startswith("checkpoint_epoch_")]
        if checkpoint_files:
            # 找到最新的檢查點
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
            latest_checkpoint = os.path.join("models", checkpoint_files[0])
            try:
                checkpoint = torch.load(latest_checkpoint, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch'] + 1
                best_string_accuracy = checkpoint.get('string_accuracy', 0)
                logger.log(f"成功從 {latest_checkpoint} 載入檢查點，從 epoch {start_epoch} 繼續訓練")
                logger.summary(f"從檢查點繼續訓練，epoch {start_epoch}")
            except Exception as e:
                logger.log(f"載入檢查點失敗: {str(e)}，從頭開始訓練")
                start_epoch = 0
                best_string_accuracy = 0

        best_char_accuracy = 0
        no_improve_count = 0
        train_losses, train_accs = [], []
        test_losses, test_string_accs, test_char_accs = [], [], []

        logger.log("開始訓練迴圈...")
        logger.summary(f"開始訓練迴圈, 最大週期={config['num_epochs']}")
        
        # 監控記憶體使用
        def get_memory_usage():
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            else:
                import psutil
                process = psutil.Process(os.getpid())
                return process.memory_info().rss / (1024 * 1024)  # MB
        
        for epoch in range(start_epoch, config['num_epochs']):
            logger.summary(f"週期 {epoch+1}/{config['num_epochs']}")

            # 訓練
            train_loss, train_acc = train(
                model, train_loader, criterion, optimizer, scheduler,
                device, epoch + 1, config['num_epochs'], logger,
                use_mixup=config['use_mixup']
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # 驗證
            test_loss, test_string_acc, test_char_acc, all_predictions, all_labels = validate(
                model, test_loader, criterion, device,
                epoch + 1, config['num_epochs'], logger
            )
            test_losses.append(test_loss)
            test_string_accs.append(test_string_acc)
            test_char_accs.append(test_char_acc)

            # 檢查是否是目前為止最好的模型
            if test_string_acc > best_string_accuracy:
                best_string_accuracy = test_string_acc
                best_char_accuracy = test_char_acc
                no_improve_count = 0

                # 保存最佳模型
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'config': config,
                    'char_set': config['char_set'],
                    'epoch': epoch,
                    'string_accuracy': test_string_acc,
                    'char_accuracy': test_char_acc
                }, "models/best_model.pth")

                logger.log(f"保存最佳模型 (字串準確率: {test_string_acc:.2f}%, 字元準確率: {test_char_acc:.2f}%)")
                logger.summary(f"新的最佳模型! 字串準確率: {test_string_acc:.2f}%, 字元準確率: {test_char_acc:.2f}%")
            else:
                no_improve_count += 1
                logger.log(f"無改善計數: {no_improve_count}/{config['early_stopping_patience']}")

            # 定期保存檢查點
            if (epoch + 1) % config['checkpoint_interval'] == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'config': config,
                    'char_set': config['char_set'],
                    'epoch': epoch,
                    'string_accuracy': test_string_acc,
                    'char_accuracy': test_char_acc,
                    'train_losses': train_losses,
                    'train_accs': train_accs,
                    'test_losses': test_losses,
                    'test_string_accs': test_string_accs,
                    'test_char_accs': test_char_accs,
                    'no_improve_count': no_improve_count
                }, f"models/checkpoint_epoch_{epoch}.pth")
                logger.log(f"保存週期 {epoch} 的檢查點")

            current_lr = optimizer.param_groups[0]['lr']
            logger.log(f"當前學習率: {current_lr:.6f}")
            logger.summary(f"學習率: {current_lr:.6f}")

            memory_usage = get_memory_usage()
            logger.summary(f"記憶體使用: {memory_usage:.2f} MB")

            # 提前停止檢查
            if no_improve_count >= config['early_stopping_patience']:
                logger.log(f"已達到提前停止條件，停止訓練")
                logger.summary(f"提前停止訓練，無改善次數: {no_improve_count}")
                break

        # 訓練結束
        logger.log("訓練完成!")
        logger.log(f"最佳字串準確率: {best_string_accuracy:.2f}%")
        logger.log(f"對應字元準確率: {best_char_accuracy:.2f}%")
        logger.summary(f"訓練完成! 最佳字串準確率: {best_string_accuracy:.2f}%, 字元準確率: {best_char_accuracy:.2f}%")

        # 進行錯誤分析
        logger.log("開始進行錯誤分析...")
        if 'all_predictions' in locals() and 'all_labels' in locals():
            analyze_errors(test_loader, all_predictions, all_labels, logger)

        # 生成預測視覺化
        logger.log("生成預測視覺化...")
        visualize_predictions(model, test_loader, config['char_set'], device, num_samples=20)

        # 繪製訓練曲線
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, label='訓練損失')
        plt.plot(test_losses, label='測試損失')
        plt.title('損失曲線')
        plt.xlabel('週期')
        plt.ylabel('損失')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(train_accs, label='訓練準確率')
        plt.plot(test_string_accs, label='測試字串準確率')
        plt.plot(test_char_accs, label='測試字元準確率')
        plt.title('準確率曲線')
        plt.xlabel('週期')
        plt.ylabel('準確率 (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/training_curves.png')
        logger.log("保存訓練曲線圖至 results/training_curves.png")
    
    except Exception as e:
        logger.log(f"訓練過程中發生錯誤: {str(e)}")
        logger.summary(f"訓練失敗: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
    finally:
        logger.log("訓練過程結束。")
        if hasattr(logger, 'close'):
            logger.close()

if __name__ == '__main__':
    main()