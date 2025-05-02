import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
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

def train(model, train_loader, criterion, optimizer, scheduler, device, epoch, total_epochs, logger, use_mixup=True, gradient_monitor=None):
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
        if gradient_monitor:
            gradient_monitor.step() # 調用step方法而不是直接調用_grad_hook

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

def load_pretrained_weights(model, pretrained_path, logger):
    try:
        logger.log(f"正在嘗試從 {pretrained_path} 載入預訓練權重...")
        pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
        
        # 記錄預訓練模型的結構
        if isinstance(pretrained_dict, dict) and 'model_state_dict' in pretrained_dict:
            # 如果是以字典格式保存的檢查點
            pretrained_state = pretrained_dict['model_state_dict']
            logger.log(f"從檢查點字典載入權重，包含keys: {list(pretrained_dict.keys())}")
            # 如果檢查點包含設定，記錄其內容
            if 'config' in pretrained_dict:
                logger.log(f"預訓練模型配置: {pretrained_dict['config']}")
            # 如果有字元集記錄
            if 'char_set' in pretrained_dict:
                logger.log(f"預訓練模型字元集: {pretrained_dict['char_set']}")
        else:
            # 如果只保存了state_dict
            pretrained_state = pretrained_dict
            logger.log(f"直接載入狀態字典")
        
        # 記錄預訓練權重的結構
        logger.log(f"預訓練模型結構：")
        for k, v in pretrained_state.items():
            if hasattr(v, 'shape'):
                logger.log(f"層: {k}, 形狀: {v.shape}")
            else:
                logger.log(f"層: {k}, 類型: {type(v)}")
        
        # 獲取當前模型的權重
        model_dict = model.state_dict()
        
        # 創建兼容字典，分別處理特徵提取器和分類器部分
        compatible_dict = {}
        for k, v in pretrained_state.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                compatible_dict[k] = v
                logger.log(f"完全匹配層: {k}, 形狀: {v.shape}")
            else:
                # 檢查是否只是層名稱不同但形狀相同
                for model_key in model_dict.keys():
                    if model_key.endswith(k.split('.')[-1]) and model_dict[model_key].shape == v.shape:
                        compatible_dict[model_key] = v
                        logger.log(f"名稱調整匹配: 預訓練層 {k} -> 模型層 {model_key}, 形狀: {v.shape}")
                        break
        
        # 如果發現特徵提取器匹配但分類器不匹配，可以只載入特徵部分
        features_loaded = sum(1 for k in compatible_dict if k.startswith('features.'))
        classifier_loaded = sum(1 for k in compatible_dict if k.startswith('classifier.'))
        
        if features_loaded > 0 and classifier_loaded == 0:
            logger.log(f"注意：只匹配到特徵提取器層，分類器層將使用隨機初始化")
        
        # 更新模型權重
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict, strict=False)
        
        # 計算載入權重的比例
        logger.log(f"成功載入 {len(compatible_dict)}/{len(model_dict)} 個相容的預訓練權重 ({len(compatible_dict)/len(model_dict)*100:.1f}%)")
        
        # 記錄層級別的載入統計
        features_count = sum(1 for k in model_dict if k.startswith('features.'))
        classifier_count = sum(1 for k in model_dict if k.startswith('classifier.'))
        features_loaded = sum(1 for k in compatible_dict if k.startswith('features.'))
        classifier_loaded = sum(1 for k in compatible_dict if k.startswith('classifier.'))
        
        logger.log(f"特徵提取器: 載入 {features_loaded}/{features_count} 層 ({features_loaded/features_count*100:.1f}%)")
        logger.log(f"分類器: 載入 {classifier_loaded}/{classifier_count} 層 ({classifier_loaded/classifier_count*100:.1f}% 如果是0可能需要微調)")
        
        return True, compatible_dict
    
    except FileNotFoundError:
        logger.log(f"預訓練模型檔案未找到: {pretrained_path}")
        return False, {}
    except Exception as e:
        logger.log(f"載入預訓練模型時發生錯誤: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        return False, {}

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

def adapt_model_for_pretrained(pretrained_path, char_set_size, captcha_length, image_height, image_width, logger):
    """
    分析預訓練模型的結構，並創建一個兼容的模型架構
    """
    try:
        # 載入預訓練模型
        pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
        
        # 確定預訓練模型的結構
        pretrained_state = pretrained_dict['model_state_dict'] if isinstance(pretrained_dict, dict) and 'model_state_dict' in pretrained_dict else pretrained_dict
        
        # 檢查是否有配置信息
        if isinstance(pretrained_dict, dict) and 'config' in pretrained_dict:
            pretrained_config = pretrained_dict['config']
            logger.log(f"偵測到預訓練模型配置: {pretrained_config}")
            
            # 根據預訓練模型配置調整當前配置
            old_height = pretrained_config.get('image_height', image_height)
            old_width = pretrained_config.get('image_width', image_width)
            old_captcha_length = pretrained_config.get('captcha_length', captcha_length)
            old_char_set_size = len(pretrained_config.get('char_set', []))
            
            logger.log(f"預訓練模型維度: 高={old_height}, 寬={old_width}, 字元長度={old_captcha_length}, 字元集大小={old_char_set_size}")
            logger.log(f"當前模型維度: 高={image_height}, 寬={image_width}, 字元長度={captcha_length}, 字元集大小={char_set_size}")
            
            # 如果維度匹配，可以創建完全兼容的模型
            if old_height == image_height and old_width == image_width:
                logger.log("圖像尺寸匹配，可以完全兼容特徵提取器")
            else:
                logger.log("圖像尺寸不匹配，可能需要調整特徵提取器輸出尺寸")

        # 分析特徵提取器結構 (更穩健的處理方式)
        feature_layers = [k for k in pretrained_state.keys() if k.startswith('features.')]
        if feature_layers:
            last_feature_layer = max(feature_layers, key=lambda x: int(x.split('.')[1]) if x.split('.')[1].isdigit() else 0)
            logger.log(f"預訓練模型最後的特徵層: {last_feature_layer}")

            # 檢查分類器第一層的輸入維度（如果存在）
            classifier_input_key = 'classifier.1.weight'
            if classifier_input_key in pretrained_state:
                input_features = pretrained_state[classifier_input_key].shape[1]
                logger.log(f"預訓練分類器輸入特徵數: {input_features}")

                # 計算特徵圖尺寸 (更穩健的計算方式)
                feature_height, feature_width = 1,1
                feature_channels = 128 # 預設通道數
                for k in reversed(feature_layers):
                    if k.endswith('.weight') and len(pretrained_state[k].shape) == 4:
                        feature_channels = pretrained_state[k].shape[0]
                        feature_height = pretrained_state[k].shape[2]
                        feature_width = pretrained_state[k].shape[3]
                        break

                expected_features = feature_height * feature_width * feature_channels
                logger.log(f"計算得到的特徵數: {expected_features} (高={feature_height}, 寬={feature_width}, 通道={feature_channels})")

                if input_features != expected_features:
                    logger.log(f"警告: 特徵數不匹配 (預訓練={input_features}, 計算得到={expected_features})")
                    logger.log("將調整分類器的輸入尺寸以匹配")

        return True
    except Exception as e:
        logger.log(f"分析預訓練模型時發生錯誤: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        return False

class GradientMonitor:
    def __init__(self, model, logger, log_interval=10):
        self.model = model
        self.logger = logger
        self.log_interval = log_interval
        self.grad_stats = {}  # Use a dictionary to store gradient statistics
        self.iter_count = 0 # Counter for iterations
        self.hooks = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.grad_stats[name] = {'mean': [], 'std': [], 'min': [], 'max': []}
                self.hooks.append(param.register_hook(lambda grad, name=name: self._grad_hook(grad, name)))

    def _grad_hook(self, grad, name):
        if grad is not None:
            self.grad_stats[name]['mean'].append(grad.abs().mean().item())
            self.grad_stats[name]['std'].append(grad.std().item())
            self.grad_stats[name]['min'].append(grad.min().item())
            self.grad_stats[name]['max'].append(grad.max().item())
        return grad

    def step(self):
        self.iter_count += 1
        if self.iter_count % self.log_interval == 0:
            self.log_gradients()

    def log_gradients(self):
        self.logger.log(f"迭代 {self.iter_count} 的梯度統計:")
        for name, stats in self.grad_stats.items():
            if not stats['mean']: continue
            mean_grad = np.mean(stats['mean'])
            self.logger.log(f"  層: {name}, 平均梯度: {mean_grad:.6f}")
            # Reset stats for next interval
            for k in stats: stats[k] = []
        self.visualize_gradients()

    
    def visualize_gradients(self):
        """將梯度統計可視化並保存為圖片"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        plt.figure(figsize=(12, 8))
        
        # 收集特徵提取器和分類器層的梯度統計
        features_layers = [name for name in self.grad_stats.keys() if name.startswith('features.')]
        classifier_layers = [name for name in self.grad_stats.keys() if name.startswith('classifier.')]
        
        # 只取有梯度記錄的層
        features_layers = [name for name in features_layers if 'mean' in self.grad_stats[name] and self.grad_stats[name]['mean']]
        classifier_layers = [name for name in classifier_layers if 'mean' in self.grad_stats[name] and self.grad_stats[name]['mean']]
        
        # 排序層名稱
        features_layers.sort(key=lambda x: int(x.split('.')[1]) if x.split('.')[1].isdigit() else 0)
        classifier_layers.sort(key=lambda x: int(x.split('.')[1]) if x.split('.')[1].isdigit() else 0)
        
        # 繪製特徵提取器梯度
        plt.subplot(2, 1, 1)
        x = np.arange(len(features_layers))
        means = [np.mean(self.grad_stats[name]['mean']) for name in features_layers]
        stds = [np.mean(self.grad_stats[name]['std']) for name in features_layers]
        
        plt.bar(x, means, yerr=stds, align='center', alpha=0.7, capsize=5)
        plt.xticks(x, [name.split('.')[-2:] for name in features_layers], rotation=45)
        plt.ylabel('平均梯度大小')
        plt.title('特徵提取器層梯度統計')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 繪製分類器梯度
        plt.subplot(2, 1, 2)
        x = np.arange(len(classifier_layers))
        means = [np.mean(self.grad_stats[name]['mean']) for name in classifier_layers]
        stds = [np.mean(self.grad_stats[name]['std']) for name in classifier_layers]
        
        plt.bar(x, means, yerr=stds, align='center', alpha=0.7, capsize=5)
        plt.xticks(x, [name.split('.')[-2:] for name in classifier_layers], rotation=45)
        plt.ylabel('平均梯度大小')
        plt.title('分類器層梯度統計')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('gradient_statistics.png')
        self.logger.log("梯度統計視覺化已保存至: gradient_statistics.png")
    
    def remove_hooks(self):
        """移除所有梯度鉤子"""
        for hook in self.hooks:
            hook.remove()
        self.logger.log("已移除梯度監控鉤子")
        
def main():
    config = {
        'train_dir': r"C:\Users\C14511\Desktop\captcha\CAPTCHA\0000",       # 訓練資料集目錄
        'test_dir': r"C:\Users\C14511\Desktop\captcha\CAPTCHA\training01",  # 測試資料集目錄
        'pretrained_model_path': r"C:\Users\C14511\captcha\best_denoised_char_model.pth", # 預訓練模型路徑 (可選)
        'image_width': 225,                                                 # 輸入圖片寬度
        'image_height': 69,                                                 # 輸入圖片高度
        'batch_size': 32,                                                   # 批次大小
        'num_epochs': 500,                                                  # 最大訓練週期
        'learning_rate': 0.001,                                             # 學習率
        'weight_decay': 0.01,                                               # 權重衰減
        'captcha_length': 6,                                                # CAPTCHA 字元長度
        'use_mixup': True,                                                  # 是否使用 Mixup 資料增強
        'use_focalloss': True,                                              # 是否使用 Focal Loss
        'gamma': 2.0,                                                       # Focal Loss 的 gamma 參數
        'use_augmentation': True,                                           # 是否使用額外資料增強
        'early_stopping_patience': 30,                                      # 提前停止的耐心值
        'checkpoint_interval': 25,                                          # 定期儲存檢查點的間隔
        'char_set': ['2', '3', '4', '5', '6', '7', '8', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'k', 'm', 'p', 'r', 'w', 'x', 'y'] # 使用的字元集
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    with open('config.json', 'w', encoding='utf-8') as f: # 加入 encoding='utf-8'
        json.dump(config, f, indent=2)

    transform = transforms.Compose([
        transforms.Resize((config['image_height'], config['image_width'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    logger = TrainingLogger(log_dir='logs')
    logger.log("開始訓練...")
    logger.summary(f"CAPTCHA辨識模型訓練開始")
    logger.summary(f"模型設定: 字元長度={config['captcha_length']}, 字元集大小={len(config['char_set'])}")

    try:
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

        train_dataset = CaptchaDataset(
            img_dir=config['train_dir'],
            char_set=config['char_set'],
            transform=train_transform,
            captcha_length=config['captcha_length']
        )
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

        logger.log(f"訓練集大小: {len(train_dataset)}")
        logger.log(f"測試集大小: {len(test_dataset)}")
        logger.summary(f"資料集: 訓練={len(train_dataset)}個樣本, 測試={len(test_dataset)}個樣本")
        logger.log(f"字元集: {train_dataset.char_set}")
        logger.log(f"字元集大小: {len(train_dataset.char_set)}")

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
        
        model = CaptchaCNN(
            num_chars=config['captcha_length'], 
            num_classes=len(train_dataset.char_set),
            image_height=config['image_height'],
            image_width=config['image_width']
        ).to(device)
        
        if config['pretrained_model_path']:
            logger.log(f"嘗試從 {config['pretrained_model_path']} 載入預訓練權重...")
            success = load_pretrained_weights(model, config['pretrained_model_path'], logger) 
            if not success:
                logger.log("載入預訓練模型失敗，使用隨機初始化的權重")
                logger.summary("使用隨機初始化的權重 (無法載入預訓練模型)")
            else:
                logger.summary(f"成功載入預訓練模型: {config['pretrained_model_path']}")
        
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

        checkpoint_string_accuracy = 0
        checkpoint_char_accuracy = 0
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
        
        for epoch in range(config['num_epochs']):
            logger.summary(f"週期 {epoch+1}/{config['num_epochs']}")

            train_loss, train_acc = train(
                model, train_loader, criterion, optimizer, scheduler,
                device, epoch + 1, config['num_epochs'], logger,
                use_mixup=config['use_mixup']
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            test_loss, test_string_acc, test_char_acc, all_predictions, all_labels = validate(
                model, test_loader, criterion, device,
                epoch + 1, config['num_epochs'], logger
            )
            test_losses.append(test_loss)
            test_string_accs.append(test_string_acc)
            test_char_accs.append(test_char_acc)

            current_lr = optimizer.param_groups[0]['lr']
            logger.log(f"當前學習率: {current_lr:.6f}")
            logger.summary(f"學習率: {current_lr:.6f}")

            memory_usage = get_memory_usage()
            logger.summary(f"記憶體使用: {memory_usage:.2f} MB")


            # Improved early stopping: considers both character and string accuracy
            improved = False
            if test_char_acc > checkpoint_char_accuracy or test_string_acc > checkpoint_string_accuracy:
                checkpoint_string_accuracy = max(checkpoint_string_accuracy, test_string_acc)
                checkpoint_char_accuracy = max(checkpoint_char_accuracy, test_char_acc)
                improved = True
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'checkpoint_string_accuracy': checkpoint_string_accuracy,
                    'checkpoint_char_accuracy': checkpoint_char_accuracy,
                    'char_set': train_dataset.char_set,
                    'config': config
                }
                torch.save(checkpoint, os.path.join('models', 'best_model.pth'))
                logger.log(f"已儲存新的最佳模型，字串準確率: {checkpoint_string_accuracy:.2f}%，字元準確率: {checkpoint_char_accuracy:.2f}%")
                logger.summary(f"新的最佳模型: 字串準確率={checkpoint_string_accuracy:.2f}%, 字元準確率={checkpoint_char_accuracy:.2f}%")
                no_improve_count = 0
                if test_string_acc > 95.0:  # Adjust threshold as needed
                    logger.log("進行錯誤分析...")
                    logger.summary("進行錯誤分析...")
                    analyze_errors(
                        test_loader, all_predictions, all_labels, logger,
                        output_dir=os.path.join('results', f'error_analysis_epoch_{epoch+1}')
                    )
            else:
                no_improve_count += 1
                logger.summary(f"無改進計數: {no_improve_count}/{config['early_stopping_patience']}")
            
            if (epoch + 1) % config['checkpoint_interval'] == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'current_accuracy': test_string_acc,
                    'checkpoint_accuracy': checkpoint_string_accuracy,
                    'char_set': train_dataset.char_set,
                    'config': config
                }
                torch.save(checkpoint, os.path.join('models', f'checkpoint_epoch_{epoch+1}.pth'))
                logger.log(f"已儲存定期檢查點 - 週期 {epoch+1}")
                logger.summary(f"儲存檢查點: 週期 {epoch+1}")
                
            if no_improve_count >= config['early_stopping_patience']:
                logger.log(f"提早停止訓練! 連續 {config['early_stopping_patience']} 個週期沒有改善。")
                logger.summary(f"連續 {config['early_stopping_patience']} 輪無改進，提前停止訓練!")
                logger.summary("")  # 空行
                break

        logger.log(f"訓練完成！最佳測試準確率: {checkpoint_string_accuracy:.2f}%")
        logger.summary(f"訓練完成！最佳字串準確率: {checkpoint_string_accuracy:.2f}%, 最佳字元準確率: {checkpoint_char_accuracy:.2f}%")
        
        plt.figure(figsize=(12, 15))
        plt.subplot(3, 1, 1)
        plt.plot(train_losses, label='訓練損失')
        plt.plot(test_losses, label='驗證損失')
        plt.xlabel('週期')
        plt.ylabel('損失')
        plt.legend()
        plt.title('損失曲線')
        
        plt.subplot(3, 1, 2)
        plt.plot(train_accs, label='訓練準確率')
        plt.plot(test_string_accs, label='字串準確率')
        plt.xlabel('週期')
        plt.ylabel('準確率 (%)')
        plt.legend()
        plt.title('字串準確率曲線')
        
        plt.subplot(3, 1, 3)
        plt.plot(test_char_accs, label='字元準確率')
        plt.xlabel('週期')
        plt.ylabel('準確率 (%)')
        plt.legend()
        plt.title('字元準確率曲線')
        
        plt.tight_layout()
        plt.savefig(os.path.join('results', 'training_curves.png'))
        logger.log("已保存訓練曲線圖")
        logger.summary("已保存訓練曲線圖")
        
        # 最後關閉日誌檔案
        logger.close()
        
    except Exception as e:
        logger.log(f"訓練過程中發生錯誤: {str(e)}")
        logger.summary(f"訓練失敗: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        logger.close()
    finally:
        logger.log("訓練過程結束。")
        if hasattr(logger, 'close'):
            logger.close()

if __name__ == '__main__':
    main()