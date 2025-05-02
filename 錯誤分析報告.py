import os
from collections import defaultdict, Counter

def analyze_errors(results_file):
    # 儲存錯誤分析的結果
    error_stats = {
        'total_errors': 0,
        'position_errors': defaultdict(int),  # 記錄每個位置的錯誤次數
        'char_confusion': defaultdict(lambda: defaultdict(int)),  # 字符混淆矩陣
        'error_samples': [],  # 儲存所有錯誤樣本
        'most_common_errors': defaultdict(int),  # 最常見的錯誤模式
        'character_errors': defaultdict(int)  # 每個字符的錯誤次數
    }
    
    current_sample = {}
    reading_sample = False
    
    # 讀取測試結果文件
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if reading_sample and current_sample:
                    # 如果是錯誤樣本，進行分析
                    if not current_sample['correct']:
                        true_label = current_sample['true_label']
                        pred_label = current_sample['predicted']
                        error_stats['total_errors'] += 1
                        error_stats['error_samples'].append(current_sample.copy())
                        
                        # 分析每個位置的錯誤
                        for pos, (true_char, pred_char) in enumerate(zip(true_label, pred_label)):
                            if true_char != pred_char:
                                error_stats['position_errors'][pos] += 1
                                error_stats['char_confusion'][true_char][pred_char] += 1
                                error_stats['character_errors'][true_char] += 1
                        
                        # 記錄錯誤模式
                        error_pattern = f"{true_label}->{pred_label}"
                        error_stats['most_common_errors'][error_pattern] += 1
                
                current_sample = {}
                reading_sample = False
                continue
                
            if line.startswith('檔案:'):
                reading_sample = True
                current_sample['filename'] = line.split(': ')[1]
            elif line.startswith('實際值:'):
                current_sample['true_label'] = line.split(': ')[1]
            elif line.startswith('預測值:'):
                current_sample['predicted'] = line.split(': ')[1]
            elif line.startswith('正確:'):
                current_sample['correct'] = line.split(': ')[1] == 'True'
    
    # 生成錯誤分析報告
    generate_error_report(error_stats)

def generate_error_report(error_stats):
    with open('error_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== 驗證碼錯誤分析報告 ===\n\n")
        
        # 1. 總體錯誤統計
        f.write("1. 總體錯誤統計\n")
        f.write(f"總錯誤數量: {error_stats['total_errors']}\n\n")
        
        # 2. 位置錯誤分析
        f.write("2. 位置錯誤分析\n")
        for pos, count in sorted(error_stats['position_errors'].items()):
            f.write(f"位置 {pos + 1}: {count} 次錯誤 ({count/error_stats['total_errors']*100:.2f}%)\n")
        f.write("\n")
        
        # 3. 最常見錯誤模式
        f.write("3. 最常見錯誤模式 (Top 10)\n")
        for pattern, count in Counter(error_stats['most_common_errors']).most_common(10):
            f.write(f"{pattern}: {count} 次\n")
        f.write("\n")
        
        # 4. 字符錯誤統計
        f.write("4. 各字符錯誤統計\n")
        for char, count in sorted(error_stats['character_errors'].items(), key=lambda x: x[1], reverse=True):
            f.write(f"字符 '{char}': {count} 次錯誤\n")
        f.write("\n")
        
        # 5. 字符混淆矩陣 (Top 10)
        f.write("5. 最常見字符混淆 (Top 10)\n")
        confusion_list = []
        for true_char, pred_dict in error_stats['char_confusion'].items():
            for pred_char, count in pred_dict.items():
                confusion_list.append((true_char, pred_char, count))
        
        for true_char, pred_char, count in sorted(confusion_list, key=lambda x: x[2], reverse=True)[:10]:
            f.write(f"'{true_char}' 被誤認為 '{pred_char}': {count} 次\n")
        f.write("\n")
        
        # 6. 詳細錯誤樣本列表
        f.write("6. 錯誤樣本詳細列表\n")
        for sample in error_stats['error_samples']:
            f.write(f"檔案: {sample['filename']}\n")
            f.write(f"實際值: {sample['true_label']}\n")
            f.write(f"預測值: {sample['predicted']}\n")
            f.write("-" * 30 + "\n")

if __name__ == '__main__':
    analyze_errors('test_results.txt')