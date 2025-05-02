import os
import numpy as np
from PIL import Image, ImageEnhance
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
from tqdm import tqdm
import random
from collections import Counter
import psutil
import gc
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

class TrainingLogger:
    def __init__(self):
        self.log_file = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    def log(self, message):
        timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        print(f"{timestamp}: {message}")  # 添加即時輸出
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{timestamp}: {message}\n")

class SingleCharDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.char_to_idx = {}
        self.idx_to_char = {}

        # 載入或創建字符映射
        mapping_file = os.path.join(root_dir, 'char_mapping.json')
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
                self.char_to_idx = mapping['char_to_idx']
                self.idx_to_char = {int(k): v for k, v in mapping['idx_to_char'].items()}

        # 遍歷每個字符目錄
        for char_dir in os.listdir(root_dir):
            char_path = os.path.join(root_dir, char_dir)
            if os.path.isdir(char_path):
                if char_dir not in self.char_to_idx:
                    idx = len(self.char_to_idx)
                    self.char_to_idx[char_dir] = idx
                    self.idx_to_char[idx] = char_dir

                # 收集該字符的所有圖片
                for img_file in os.listdir(char_path):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((
                            os.path.join(char_path, img_file),
                            self.char_to_idx[char_dir]
                        ))

        # 保存字符映射
        if not os.path.exists(mapping_file):
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'char_to_idx': self.char_to_idx,
                    'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()}
                }, f, ensure_ascii=False, indent=2)

        # 資料集分割
        if split != 'all':
            random.seed(42)  # 確保可重現性
            indices = list(range(len(self.samples)))
            random.shuffle(indices)
            train_size = int(0.8 * len(indices))  # 80/20 分割
            if split == 'train':
                self.samples = [self.samples[i] for i in indices[:train_size]]
            else:  # 'val'
                self.samples = [self.samples[i] for i in indices[train_size:]]
    
    def __len__(self):
        """返回數據集的樣本數量"""
        return len(self.samples)

    def preprocess_image(self, image):
        # 轉換為灰度圖
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 自適應二值化，對噪聲和不均勻照明更穩健
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)

        # 去除小噪點
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 膨脹主要字符，使字符更連續
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

        # 找出所有連通區域
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # 如果找到連通區域
        if contours:
            # 按面積排序，選最大的區域（假設主要字符是最大的）
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_contour = contours[0]
        
            # 獲取邊界框
            x, y, w, h = cv2.boundingRect(largest_contour)
        
            # 加入適當邊距，但不要太大以避免包含其他字符
            margin = min(5, min(x, y, binary.shape[1]-x-w, binary.shape[0]-y-h))
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(binary.shape[1] - x, w + 2 * margin)
            h = min(binary.shape[0] - y, h + 2 * margin)
        
            # 裁剪圖像
            cropped = binary[y:y+h, x:x+w]
        
            # 調整到固定大小
            resized = cv2.resize(cropped, (37, 69), interpolation=cv2.INTER_AREA)
        
            # 轉回 RGB (反轉黑白，使字符為黑色)
            resized = cv2.bitwise_not(resized)
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            # 如果沒有找到連通區域，返回原始圖像
            rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(rgb)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        # 預處理圖片
        image = np.array(image)
        image = self.preprocess_image(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)

# 修改後的單字符CNN模型
class SingleCharCNN(nn.Module):
    def __init__(self, num_classes, image_height=69, image_width=37, dropout_rate=0.3):
        super(SingleCharCNN, self).__init__()
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
            nn.Linear(512, num_classes)
        )

        self.num_chars = 1
        self.num_classes = num_classes
        self._initialize_weights()
        
        # 初始化權重
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
    
    # 修改為與多字符模型相同的輸出格式
    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        x = x.view(batch_size, 1, self.num_classes)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 修改後的訓練函數
def train_single_char(model, train_loader, criterion, optimizer, device, update_every=1, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # 確保標籤形狀正確 [batch_size, 1]
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1)

        # 前向傳播
        outputs = model(images)

        # 處理模型輸出的形狀 [batch_size, 1, num_classes]
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

        # 除以累積步數以正規化梯度
        loss = loss / update_every

        # 反向傳播
        loss.backward()

        # 每update_every批次更新一次權重
        if (batch_idx + 1) % update_every == 0 or (batch_idx + 1) == len(train_loader):
            # 梯度裁剪以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

        # 統計資料
        running_loss += loss.item() * update_every  # 還原真實loss值
        _, predicted = torch.max(outputs.data, 2)  # 修改為沿第2維度取最大值
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新進度條
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })

        # 釋放記憶體
        del images, labels, outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return running_loss / len(train_loader), 100 * correct / total

# 修改後的驗證函數
def validate_single_char(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc="Validating")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # 確保標籤形狀正確 [batch_size, 1]
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1)
                
            # 前向傳播
            outputs = model(images)
            
            # 處理模型輸出的形狀 [batch_size, 1, num_classes]
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            # 計算統計數據
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 2)  # 修改為沿第2維度取最大值
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新進度條
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
            
            # 釋放記憶體
            del images, labels, outputs, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
    return running_loss / len(val_loader), 100 * correct / total

# 修改後的錯誤分析函數
def analyze_errors(model, val_loader, idx_to_char, device, num_samples=10):
    model.eval()
    errors = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 確保標籤形狀正確
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1)
                
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 2)  # 修改為沿第2維度取最大值
            
            # 找出預測錯誤的樣本
            incorrect_mask = (predicted != labels)
            if incorrect_mask.sum().item() > 0:
                incorrect_indices = incorrect_mask.nonzero(as_tuple=True)[0]
                for idx in incorrect_indices:
                    if len(errors) < num_samples:
                        errors.append({
                            'image': images[idx].cpu(),
                            'actual': idx_to_char[labels[idx][0].item()],  # 注意索引方式的變化
                            'predicted': idx_to_char[predicted[idx][0].item()],  # 注意索引方式的變化
                            'confidence': torch.softmax(outputs[idx][0], dim=0)[predicted[idx][0]].item()  # 注意索引方式的變化
                        })
                    else:
                        break
            
            # 釋放記憶體
            del images, labels, outputs, predicted
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            if len(errors) >= num_samples:
                break
    
    return errors

# 修改後的儲存檢查點函數
def save_checkpoint(epoch, model, optimizer, best_accuracy, char_to_idx, char_set, filename):
    config = {
        'image_height': 69,
        'image_width': 37,
        'dropout_rate': 0.3,
        'char_set': char_set # 新增儲存字元集資訊
    }

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy,
        'char_to_idx': char_to_idx,
        'char_set': char_set, # 新增儲存字元集資訊
        'config': config
    }

    torch.save(checkpoint, filename)

def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    char_to_idx = checkpoint['char_to_idx']
    config = checkpoint.get('config', None)  # 獲取配置，如果沒有則為None
    return epoch, best_accuracy, char_to_idx, config

def preprocess_and_show_sample(dataset, idx=0):
    """顯示資料集中的範例圖片及其預處理效果"""
    import matplotlib.pyplot as plt
    
    img_path, label = dataset.samples[idx]
    original_img = Image.open(img_path).convert('RGB')
    
    # 預處理前的圖片
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("原始圖片")
    plt.imshow(original_img)
    plt.axis('off')
    
    # 預處理後的圖片 (但不經過轉換)
    processed_img = dataset.preprocess_image(np.array(original_img))
    plt.subplot(1, 3, 2)
    plt.title("預處理後")
    plt.imshow(processed_img)
    plt.axis('off')
    
    # 若有轉換，顯示轉換後的圖片
    if dataset.transform:
        transformed_img = dataset.transform(processed_img)
        transformed_img = transformed_img.permute(1, 2, 0).numpy()
        transformed_img = ((transformed_img * 0.5) + 0.5)  # 反標準化
        plt.subplot(1, 3, 3)
        plt.title("轉換後")
        plt.imshow(transformed_img)
        plt.axis('off')
    
    plt.savefig("sample_preprocessing.png")
    plt.close()
    
    return "sample_preprocessing.png"

def print_model_structure_and_weights(model, logger):
    """打印模型結構和預訓練權重摘要"""
    # 打印模型結構
    logger.log("\n模型結構:")
    logger.log(str(model))
    
    # 打印模型層的形狀和參數摘要
    logger.log("\n模型層參數摘要:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.log(f"層: {name}, 形狀: {param.shape}, 參數數量: {param.numel()}")
            # 打印權重的基本統計信息
            if 'weight' in name:
                logger.log(f"  平均值: {param.data.mean().item():.6f}")
                logger.log(f"  標準差: {param.data.std().item():.6f}")
                logger.log(f"  最小值: {param.data.min().item():.6f}")
                logger.log(f"  最大值: {param.data.max().item():.6f}")
    
    # 統計不同類型層的權重
    conv_weights = [param for name, param in model.named_parameters() if 'conv' in name.lower() and 'weight' in name.lower()]
    bn_weights = [param for name, param in model.named_parameters() if 'bn' in name.lower() and 'weight' in name.lower()]
    linear_weights = [param for name, param in model.named_parameters() if 'linear' in name.lower() and 'weight' in name.lower()]
    
    if conv_weights:
        concat_conv = torch.cat([w.flatten() for w in conv_weights])
        logger.log(f"\n捲積層權重統計 (總數: {len(conv_weights)}層):")
        logger.log(f"  平均值: {concat_conv.mean().item():.6f}")
        logger.log(f"  標準差: {concat_conv.std().item():.6f}")
    
    if bn_weights:
        concat_bn = torch.cat([w.flatten() for w in bn_weights])
        logger.log(f"\n批標準化層權重統計 (總數: {len(bn_weights)}層):")
        logger.log(f"  平均值: {concat_bn.mean().item():.6f}")
        logger.log(f"  標準差: {concat_bn.std().item():.6f}")
    
    if linear_weights:
        concat_linear = torch.cat([w.flatten() for w in linear_weights])
        logger.log(f"\n全連接層權重統計 (總數: {len(linear_weights)}層):")
        logger.log(f"  平均值: {concat_linear.mean().item():.6f}")
        logger.log(f"  標準差: {concat_linear.std().item():.6f}")


# 新增權重視覺化函數
def visualize_weights(model, save_path='weight_visualization.png'):
    """視覺化模型各層的權重分佈"""
    plt.figure(figsize=(15, 10))
    
    # 收集各類型層的權重
    layers_weights = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            layer_type = name.split('.')[0] if '.' in name else 'other'
            if layer_type not in layers_weights:
                layers_weights[layer_type] = []
            layers_weights[layer_type].append(param.data.cpu().numpy().flatten())
    
    # 繪製直方圖
    num_plots = len(layers_weights)
    cols = 2
    rows = (num_plots + 1) // cols
    
    for i, (layer_type, weights) in enumerate(layers_weights.items(), 1):
        plt.subplot(rows, cols, i)
        
        if weights:
            # 合併該類型所有層的權重
            all_weights = np.concatenate(weights)
            plt.hist(all_weights, bins=50, alpha=0.7)
            plt.title(f'{layer_type} 權重分佈')
            plt.xlabel('權重值')
            plt.ylabel('頻率')
            plt.grid(True, alpha=0.3)
            
            # 添加統計信息
            mean_val = np.mean(all_weights)
            std_val = np.std(all_weights)
            plt.annotate(f'平均值: {mean_val:.4f}\n標準差: {std_val:.4f}', 
                         xy=(0.05, 0.95), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path)
    return save_path

# 監控梯度的函數
def add_gradient_monitoring(model, logger, epoch_interval=5):
    """添加模型梯度監控功能"""
    gradient_stats = {}
    
    # 註冊鉤子函數收集梯度信息
    def hook_fn(name):
        def hook(grad):
            if epoch_interval > 0 and hasattr(model, 'current_epoch') and model.current_epoch % epoch_interval == 0:
                if name not in gradient_stats:
                    gradient_stats[name] = []
                # 計算梯度統計數據
                if grad is not None:
                    grad_mean = grad.mean().item()
                    grad_std = grad.std().item()
                    grad_abs_mean = grad.abs().mean().item()
                    gradient_stats[name].append({
                        'epoch': model.current_epoch,
                        'mean': grad_mean,
                        'std': grad_std,
                        'abs_mean': grad_abs_mean
                    })
                    logger.log(f"第 {model.current_epoch} 輪 - 層 {name} 梯度統計: "
                               f"平均值={grad_mean:.6f}, 標準差={grad_std:.6f}, 絕對平均值={grad_abs_mean:.6f}")
        return hook
    
    # 為模型參數註冊梯度鉤子
    hooks = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            hook = param.register_hook(hook_fn(name))
            hooks.append(hook)
    
    # 添加梯度統計視覺化函數
    def visualize_gradients(save_path='gradient_stats.png'):
        if not gradient_stats:
            logger.log("沒有收集到梯度統計信息")
            return
        
        # 為每一層繪製梯度變化圖
        plt.figure(figsize=(15, 10))
        num_layers = len(gradient_stats)
        cols = 2
        rows = (num_layers + 1) // cols
        
        for i, (name, stats) in enumerate(gradient_stats.items(), 1):
            if not stats:
                continue
                
            # 提取數據
            epochs = [stat['epoch'] for stat in stats]
            means = [stat['mean'] for stat in stats]
            stds = [stat['std'] for stat in stats]
            abs_means = [stat['abs_mean'] for stat in stats]
            
            plt.subplot(rows, cols, i)
            plt.plot(epochs, means, 'b-', label='平均值')
            plt.plot(epochs, abs_means, 'r-', label='絕對平均值')
            plt.fill_between(epochs, 
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.2, color='b')
            plt.title(f'層 {name} 梯度變化')
            plt.xlabel('輪次')
            plt.ylabel('梯度值')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path)
        logger.log(f"梯度統計視覺化已保存至: {save_path}")
        return save_path
    
    # 為模型添加移除鉤子的方法
    def remove_hooks():
        for hook in hooks:
            hook.remove()
        logger.log("已移除梯度監控鉤子")
    
    # 為模型添加新屬性和方法
    model.current_epoch = 0
    model.gradient_stats = gradient_stats
    model.visualize_gradients = visualize_gradients
    model.remove_gradient_hooks = remove_hooks
    
    # 修改train_single_char函數以更新current_epoch
    original_train = train_single_char
    def train_with_epoch_tracking(*args, **kwargs):
        model = args[0]
        model.current_epoch += 1
        return original_train(*args, **kwargs)
    
    return train_with_epoch_tracking

def main():
    # 設置資源限制
    torch.set_num_threads(2)  # 限制CPU線程數
    torch.backends.cudnn.benchmark = False  # CPU模式關閉cudnn加速
    
    # 配置參數
    single_char_dir = r"C:\Users\C14511\Desktop\captcha\CAPTCHA\0"  # 請修改為您的資料集路徑
    batch_size = 8  # 更小的批量大小
    single_char_epochs = 500  # 降低輪數以加快訓練
    learning_rate = 0.0003  # 降低學習率
    early_stopping_patience = 10  # 提前停止耐心度
    checkpoint_interval = 10  # 更頻繁保存檢查點
    update_every = 4  # 梯度累積步數
    
    # 定義資料轉換
    transform = transforms.Compose([
        transforms.Resize((69, 37)),
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((69, 37)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 初始化日誌記錄器
    logger = TrainingLogger()
    logger.log("開始訓練...")
    
    try:
        # 創建單字符數據集
        single_char_train_dataset = SingleCharDataset(single_char_dir, transform=transform, split='train')
        single_char_val_dataset = SingleCharDataset(single_char_dir, transform=val_transform, split='val')
        
        # 顯示樣本預處理效果
        logger.log("正在生成範例預處理圖片...")
        sample_img_path = preprocess_and_show_sample(single_char_train_dataset)
        logger.log(f"範例圖片已保存至 {sample_img_path}")
        
        # 記錄數據集信息
        logger.log(f"數據集信息:")
        logger.log(f"- 訓練集大小: {len(single_char_train_dataset)}")
        logger.log(f"- 驗證集大小: {len(single_char_val_dataset)}")
        logger.log(f"- 字符類別數: {len(single_char_train_dataset.char_to_idx)}")
        logger.log(f"- 字符集: {single_char_train_dataset.char_to_idx}")
        
        single_char_train_loader = DataLoader(
            single_char_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        single_char_val_loader = DataLoader(
            single_char_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        device = torch.device('cpu')
        logger.log(f"使用設備: {device}")
        
        # 使用修改後的模型
        logger.log("初始化單字符模型...")
        single_char_model = SingleCharCNN(
            num_classes=len(single_char_train_dataset.char_to_idx),
            image_height=69,
            image_width=37,
            dropout_rate=0.3
        ).to(device)
        logger.log(f"模型參數數量: {count_parameters(single_char_model):,}")
        
        # 打印初始化後的模型結構和權重
        logger.log("\n===== 初始化模型結構和權重 =====")
        print_model_structure_and_weights(single_char_model, logger)
        visualize_weights(single_char_model, 'initial_weights.png')
        logger.log(f"初始權重視覺化已保存至 initial_weights.png")

        # 使用標準交叉熵損失函數
        criterion = nn.CrossEntropyLoss()
        
        # 使用帶有權重衰減的AdamW優化器
        optimizer = optim.AdamW(single_char_model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # 使用餘弦衰減學習率
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=single_char_epochs)
        
        best_single_char_accuracy = 0
        no_improve_count = 0

        # 添加梯度監控 (每5個epoch記錄一次)
        logger.log("正在設置梯度監控...")
        train_single_char_with_monitoring = add_gradient_monitoring(single_char_model, logger, epoch_interval=20)

        logger.log("開始訓練循環...")
        for epoch in range(single_char_epochs):
            # 手動執行垃圾回收
            gc.collect()
            
            logger.log(f"\n單字符預訓練週期 {epoch+1}/{single_char_epochs}")
            
            # 使用梯度累積進行訓練
            train_loss, train_acc = train_single_char_with_monitoring( # 使用監控版本的訓練函數
                single_char_model, 
                single_char_train_loader, 
                criterion, 
                optimizer, 
                device, 
                update_every=update_every
            )
            
            val_loss, val_acc = validate_single_char(
                single_char_model, 
                single_char_val_loader, 
                criterion, 
                device
            )
            
            # 更新學習率
            scheduler.step()
            
            # 記錄當前學習率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 輸出訓練統計
            logger.log(f"訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.2f}%")
            logger.log(f"驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.2f}%")
            logger.log(f"學習率: {current_lr:.6f}")
            logger.log(f"記憶體使用: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
            
            # 保存檢查點
            if val_acc > best_single_char_accuracy:
                best_single_char_accuracy = val_acc
                no_improve_count = 0
                save_checkpoint(
                    epoch, 
                    single_char_model, 
                    optimizer, 
                    best_single_char_accuracy, 
                    single_char_train_dataset.char_to_idx, 
                    list(single_char_train_dataset.char_to_idx.keys()),  # 添加 char_set 參數
                    'best_single_char_model.pth'  # 添加 filename 參數
                )
                logger.log(f"已儲存新的最佳單字符模型，準確率: {best_single_char_accuracy:.2f}%")
                
                # 在保存最佳模型時記錄當前權重
                logger.log("\n===== 最佳模型的權重信息 =====")
                print_model_structure_and_weights(single_char_model, logger)
                visualize_weights(single_char_model, 'best_model_weights.png')
                logger.log(f"最佳模型權重視覺化已保存至 best_model_weights.png")
            else:
                no_improve_count += 1
                logger.log(f"無改進計數: {no_improve_count}/{early_stopping_patience}")
            
            # 定期保存
            save_checkpoint(
                    epoch, 
                    single_char_model, 
                    optimizer, 
                    best_single_char_accuracy, 
                    single_char_train_dataset.char_to_idx, 
                    list(single_char_train_dataset.char_to_idx.keys()),  # 添加 char_set 參數
                    'best_single_char_model.pth'  # 添加 filename 參數
                )
            
            # 提前停止
            if no_improve_count >= early_stopping_patience:
                logger.log(f"連續 {early_stopping_patience} 輪無改進，提前停止訓練!")
                break
        
        logger.log(f"\n單字符預訓練完成！最佳驗證準確率: {best_single_char_accuracy:.2f}%")
        
        # 再次查看最終模型的權重
        logger.log("\n===== 最終模型結構和權重 =====")
        print_model_structure_and_weights(single_char_model, logger)
        visualize_weights(single_char_model, 'final_weights00.png')
        logger.log(f"最終權重視覺化已保存至 final_weights00.png")
        
        # 進行錯誤分析
        logger.log("\n進行錯誤分析...")
        errors = analyze_errors(
            single_char_model, 
            single_char_val_loader, 
            {v: k for k, v in single_char_train_dataset.char_to_idx.items()}, 
            device, 
            num_samples=10
        )
        
        if errors:
            # 儲存錯誤案例圖片
            fig, axes = plt.subplots(2, 5, figsize=(15, 6)) if len(errors) > 5 else plt.subplots(1, len(errors), figsize=(15, 3))
            axes = axes.flatten() if len(errors) > 1 else [axes]
            
            for i, error in enumerate(errors):
                if i < len(axes):
                    img = error['image'].permute(1, 2, 0).numpy()
                    img = ((img * 0.5) + 0.5) * 255  # 反標準化
                    img = img.astype(np.uint8)
                    axes[i].imshow(img)
                    axes[i].set_title(f"實際: {error['actual']}, 預測: {error['predicted']}")
                    axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig('error_analysis00.png')
            logger.log("錯誤分析完成並保存到error_analysis00.png")
        
        # 在訓練結束後，添加以下代碼
        gradient_vis_path = single_char_model.visualize_gradients('gradient_statistics00.png')
        logger.log(f"梯度統計視覺化已保存至: {gradient_vis_path}")
        single_char_model.remove_gradient_hooks()
        
    except Exception as e:
        logger.log(f"訓練過程中發生錯誤: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        raise

if __name__ == '__main__':
    main()