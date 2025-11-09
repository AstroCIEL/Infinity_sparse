# tools/calculate_clip_score_offline.py
import json
import torch
import torch.nn as nn
import numpy as np
import os
import os.path as osp
import argparse
from PIL import Image
import clip
import time
import matplotlib.pyplot as plt


def load_clip_model(model_name="ViT-B/32", device="cuda"):
    """加载CLIP模型"""
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess

def calculate_clip_score_for_batch(model, preprocess, image_paths, texts, device="cuda"):
    """批量计算CLIP Score"""
    # 预处理图像
    images = []
    valid_indices = []
    
    for i, image_path in enumerate(image_paths):
        try:
            if osp.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                image_tensor = preprocess(image).unsqueeze(0)
                images.append(image_tensor)
                valid_indices.append(i)
            else:
                print(f"Warning: Image not found: {image_path}")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
    
    if not images:
        return [], []
    
    # 批量处理图像
    image_batch = torch.cat(images).to(device)
    text_inputs = clip.tokenize([texts[i] for i in valid_indices]).to(device)
    
    # 计算特征
    with torch.no_grad():
        image_features = model.encode_image(image_batch)
        text_features = model.encode_text(text_inputs)
    
    # 归一化特征
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 计算余弦相似度
    similarities = (image_features * text_features).sum(dim=1).cpu().numpy()
    
    return similarities.tolist(), valid_indices

def plot_clip_score_distribution(json_file_path=None, json_string=None, output_file_name=None):
    """
    解析JSON文件/字符串，提取clip_score，并绘制直方图。

    :param json_file_path: JSON文件的路径。
    :param json_string: 包含JSON数据的字符串（用于测试）。
    """
    
    # 1. 加载JSON数据
    if json_file_path:
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"错误：找不到文件 {json_file_path}")
            return
        except json.JSONDecodeError:
            print(f"错误：文件 {json_file_path} 不是有效的JSON格式")
            return
    elif json_string:
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError:
            print("错误：提供的JSON字符串不是有效的JSON格式")
            return
    else:
        print("错误：必须提供文件路径或JSON字符串")
        return

    # 2. 提取 clip_score 值
    clip_scores = []
    if "detailed_results" in data and isinstance(data["detailed_results"], list):
        for item in data["detailed_results"]:
            if "clip_score" in item:
                clip_scores.append(item["clip_score"])

    if not clip_scores:
        print("未找到任何 clip_score 数据。")
        return

    # 将列表转换为 NumPy 数组以便绘图
    scores_array = np.array(clip_scores)

    # 3. 绘制数据分布图 (直方图)
    
    # 获取统计信息以便在图上显示
    mean_score = scores_array.mean()
    std_score = scores_array.std()
    
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    # bins='auto' 让 Matplotlib 自动选择合适的 bin 数量
    plt.hist(scores_array, bins=30, color='#1f77b4', edgecolor='black', alpha=0.7)
    
    # 添加平均值线
    plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean Score: {mean_score:.4f}')

    # 添加标题和标签
    plt.title('CLIP Score Distribution', fontsize=16)
    plt.xlabel('CLIP Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5, linestyle='--')
    
    # 添加统计信息文本
    text_info = (f"Total Samples: {len(scores_array)}\n"
                 f"Mean: {mean_score:.4f}\n"
                 f"Std Dev: {std_score:.4f}")
    plt.text(0.95, 0.95, text_info, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='right', 
             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.6))

    # 显示图形
    plt.savefig(output_file_name, bbox_inches='tight', dpi=300)
    print(f"图表已保存到: {output_file_name}")
    plt.close() # 关闭图形，释放内存

def main():
    parser = argparse.ArgumentParser(description='Calculate CLIP scores for generated images')
    parser.add_argument('--generation_dir', type=str, required=True, help='Directory containing generated images and metadata')
    parser.add_argument('--clip_model_name', type=str, default='ViT-B/32', 
                       choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for CLIP inference')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID for CLIP evaluation')
    
    args = parser.parse_args()
    
    # 加载CLIP模型
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Loading CLIP model {args.clip_model_name} on {device}")
    clip_model, clip_preprocess = load_clip_model(args.clip_model_name, device)
    
    # 收集所有元数据文件
    metadata_files = []
    for fname in os.listdir(args.generation_dir):
        if fname.startswith('generation_metadata_gpu') and fname.endswith('.json'):
            metadata_files.append(osp.join(args.generation_dir, fname))
    
    if not metadata_files:
        print("No metadata files found. Checking for completion files...")
        # 检查生成是否完成
        completion_files = [f for f in os.listdir(args.generation_dir) if f.endswith('.done')]
        if not completion_files:
            print("Error: No generation metadata or completion files found.")
            return
    
    # 加载所有元数据
    all_metadata = []
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                all_metadata.extend(metadata)
            print(f"Loaded {len(metadata)} items from {metadata_file}")
        except Exception as e:
            print(f"Error loading {metadata_file}: {e}")
    
    if not all_metadata:
        print("No metadata found. Please check if image generation completed successfully.")
        return
    
    print(f"Total images to evaluate: {len(all_metadata)}")
    
    # 准备批量处理
    image_paths = [item['image_path'] for item in all_metadata]
    texts = [item['prompt'] for item in all_metadata]
    
    # 批量计算CLIP Score
    all_scores = [None] * len(all_metadata)  # 预分配列表
    
    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i + args.batch_size]
        batch_texts = texts[i:i + args.batch_size]
        
        batch_scores, valid_indices = calculate_clip_score_for_batch(
            clip_model, clip_preprocess, batch_paths, batch_texts, device
        )
        
        # 将分数放回正确位置
        for score_idx, original_idx in enumerate(valid_indices):
            all_scores[i + original_idx] = batch_scores[score_idx]
        
        print(f"Processed batch {i//args.batch_size + 1}/{(len(image_paths)-1)//args.batch_size + 1}")
    
    # 过滤掉失败的项目
    valid_results = []
    for i, (metadata, score) in enumerate(zip(all_metadata, all_scores)):
        if score is not None:
            result = metadata.copy()
            result['clip_score'] = score
            valid_results.append(result)
    
    print(f"Successfully evaluated {len(valid_results)}/{len(all_metadata)} images")
    
    # 计算统计信息
    if valid_results:
        scores = [r['clip_score'] for r in valid_results]
        stats = {
            'total_images': len(valid_results),
            'mean_clip_score': np.mean(scores),
            'std_clip_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'median_score': np.median(scores),
            'clip_model': args.clip_model_name,
            'evaluation_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存结果
        results_file = osp.join(args.generation_dir, 'clip_scores_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'statistics': stats,
                'detailed_results': valid_results
            }, f, indent=2)
        
        # 打印统计信息
        print("\n" + "="*50)
        print("CLIP Score Evaluation Results")
        print("="*50)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("="*50)
        
        # 保存简化的统计文件
        stats_file = osp.join(args.generation_dir, 'clip_score_statistics.txt')
        with open(stats_file, 'w') as f:
            f.write("CLIP Score Statistics\n")
            f.write("====================\n")
            f.write(f"Model: {args.clip_model_name}\n")
            f.write(f"Total images: {stats['total_images']}\n")
            f.write(f"Mean CLIP Score: {stats['mean_clip_score']:.4f}\n")
            f.write(f"Std CLIP Score: {stats['std_clip_score']:.4f}\n")
            f.write(f"Min Score: {stats['min_score']:.4f}\n")
            f.write(f"Max Score: {stats['max_score']:.4f}\n")
            f.write(f"Median Score: {stats['median_score']:.4f}\n")
            f.write(f"Evaluation Time: {stats['evaluation_time']}\n")
            
        plot_clip_score_distribution(results_file,output_file_name=osp.join(args.generation_dir, 'clip_score_statistics.png'))
    else:
        print("No valid results to report.")
        

if __name__ == '__main__':
    main()