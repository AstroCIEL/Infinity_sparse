import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Optional
import os

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Optional
import os

class SparsityMaskAnalyzer:
    def __init__(self, pkl_path: str):
        """
        初始化分析器
        
        Args:
            pkl_path: sparsity mask的pkl文件路径
        """
        self.pkl_path = pkl_path
        self.mask_data = None
        self.analysis_results = {}
        self.load_data()
    
    def load_data(self):
        """加载pkl文件数据"""
        try:
            with open(self.pkl_path, 'rb') as f:
                self.mask_data = pickle.load(f)
            print(f"成功加载数据: {self.pkl_path}")
            print(f"总stage数: {self.mask_data['metadata']['total_stages']}")
            print(f"总block数: {self.mask_data['metadata']['total_blocks']}")
            print(f"稀疏比例: {self.mask_data['metadata']['sparsity_ratio']}")
        except Exception as e:
            print(f"加载数据失败: {e}")
            self.mask_data = None
    
    def get_stage_block_info(self) -> List[Dict[str, Any]]:
        """获取所有stage和block的基本信息"""
        if self.mask_data is None:
            return []
        
        info_list = []
        for key, mask_dict in self.mask_data['masks'].items():
            # 解析key: stage_{si}_block_{bi}
            parts = key.split('_')
            stage_idx = int(parts[1])
            block_idx = int(parts[3])
            
            for attn_type, attn_data in mask_dict.items():
                info = {
                    'stage': stage_idx,
                    'block': block_idx,
                    'attention_type': attn_type,
                    'mask_shape': attn_data['shape'],
                    'sparsity_ratio': attn_data['sparsity_ratio'],
                    'dtype': attn_data['dtype'],
                    'key': key
                }
                info_list.append(info)
        
        return info_list
    
    def analyze_overall_sparsity(self) -> Dict[str, Any]:
        """分析整体稀疏性统计"""
        if self.mask_data is None:
            return {}
        
        info_list = self.get_stage_block_info()
        if not info_list:
            return {}
        
        df = pd.DataFrame(info_list)
        
        # 基本统计
        total_masks = len(info_list)
        avg_sparsity = df['sparsity_ratio'].mean()
        std_sparsity = df['sparsity_ratio'].std()
        min_sparsity = df['sparsity_ratio'].min()
        max_sparsity = df['sparsity_ratio'].max()
        
        # 按attention类型分组统计
        type_stats = df.groupby('attention_type')['sparsity_ratio'].agg(['mean', 'std', 'count']).to_dict()
        
        # 按stage分组统计
        stage_stats = df.groupby('stage')['sparsity_ratio'].agg(['mean', 'std', 'count']).to_dict()
        
        results = {
            'total_masks': total_masks,
            'avg_sparsity': avg_sparsity,
            'std_sparsity': std_sparsity,
            'min_sparsity': min_sparsity,
            'max_sparsity': max_sparsity,
            'type_stats': type_stats,
            'stage_stats': stage_stats,
            'dataframe': df
        }
        
        self.analysis_results['overall'] = results
        return results
    
    def analyze_head_sparsity(self, stage: int, block: int, attention_type: str) -> Dict[str, Any]:
        """分析特定stage、block、attention类型的head-wise稀疏性"""
        key = f"stage_{stage}_block_{block}"
        
        if (self.mask_data is None or 
            key not in self.mask_data['masks'] or 
            attention_type not in self.mask_data['masks'][key]):
            print(f"未找到数据: {key}, {attention_type}")
            return {}
        
        mask_data = self.mask_data['masks'][key][attention_type]
        mask_tensor = mask_data['mask']
        
        if mask_tensor.dim() != 3:  # 应该是 (num_heads, seq_len_q, seq_len_k)
            print(f"Mask维度异常: {mask_tensor.shape}")
            return {}
        
        num_heads = mask_tensor.shape[0]
        head_sparsity = []
        
        # 计算每个head的稀疏度
        for head_idx in range(num_heads):
            head_mask = mask_tensor[head_idx]
            sparsity = head_mask.float().mean().item()
            head_sparsity.append(sparsity)
        
        # 统计信息
        results = {
            'stage': stage,
            'block': block,
            'attention_type': attention_type,
            'num_heads': num_heads,
            'head_sparsity': head_sparsity,
            'avg_head_sparsity': np.mean(head_sparsity),
            'std_head_sparsity': np.std(head_sparsity),
            'min_head_sparsity': np.min(head_sparsity),
            'max_head_sparsity': np.max(head_sparsity),
            'mask_shape': mask_tensor.shape
        }
        
        # 保存到分析结果
        result_key = f"stage_{stage}_block_{block}_{attention_type}"
        self.analysis_results[result_key] = results
        
        return results
    
    def visualize_head_sparsity(self, stage: int, block: int, attention_type: str, 
                               save_path: Optional[str] = None):
        """可视化特定mask的head-wise稀疏性"""
        results = self.analyze_head_sparsity(stage, block, attention_type)
        if not results:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 子图1: 每个head的稀疏度柱状图
        head_indices = range(results['num_heads'])
        ax1.bar(head_indices, results['head_sparsity'])
        ax1.set_xlabel('Head Index')
        ax1.set_ylabel('Sparsity Ratio')
        ax1.set_title(f'Head-wise Sparsity\nStage {stage}, Block {block}, {attention_type}')
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 稀疏度分布直方图
        ax2.hist(results['head_sparsity'], bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(results['avg_head_sparsity'], color='red', linestyle='--', 
                    label=f'Avg: {results["avg_head_sparsity"]:.3f}')
        ax2.set_xlabel('Sparsity Ratio')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Sparsity Distribution Across Heads')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"保存图像到: {save_path}")
        
        plt.show()
    
    def visualize_mask_heatmap(self, stage: int, block: int, attention_type: str, 
                             head_idx: int = 0, save_path: Optional[str] = None):
        """可视化特定head的mask热力图"""
        key = f"stage_{stage}_block_{block}"
        
        if (self.mask_data is None or 
            key not in self.mask_data['masks'] or 
            attention_type not in self.mask_data['masks'][key]):
            print(f"未找到数据: {key}, {attention_type}")
            return
        
        mask_data = self.mask_data['masks'][key][attention_type]
        mask_tensor = mask_data['mask']
        
        if head_idx >= mask_tensor.shape[0]:
            print(f"Head索引超出范围: {head_idx} >= {mask_tensor.shape[0]}")
            return
        
        # 获取特定head的mask
        head_mask = mask_tensor[head_idx].numpy()
        
        # 检查mask的维度，如果是3D则压缩为2D
        if head_mask.ndim == 3:
            # 压缩为2D，保留前两个维度
            head_mask = head_mask.squeeze()
        
        # 如果压缩后仍然是1D，则重塑为2D
        if head_mask.ndim == 1:
            # 如果是1D，尝试重塑为正方形或矩形
            seq_len = head_mask.shape[0]
            # 找到最接近的正方形尺寸
            sqrt_len = int(np.sqrt(seq_len))
            if sqrt_len * sqrt_len == seq_len:
                # 完美正方形
                head_mask = head_mask.reshape(sqrt_len, sqrt_len)
            else:
                # 尝试找到合适的矩形尺寸
                factors = []
                for i in range(1, int(np.sqrt(seq_len)) + 1):
                    if seq_len % i == 0:
                        factors.append((i, seq_len // i))
                if factors:
                    # 使用最大的因子对
                    h, w = factors[-1]
                    head_mask = head_mask.reshape(h, w)
                else:
                    # 无法重塑，使用条形图替代
                    print(f"无法将1D mask (长度={seq_len}) 重塑为2D，使用条形图替代")
                    self._plot_1d_mask_as_bar(head_mask, stage, block, attention_type, head_idx, save_path)
                    return
        
        # 如果仍然是1D，使用条形图
        if head_mask.ndim == 1:
            self._plot_1d_mask_as_bar(head_mask, stage, block, attention_type, head_idx, save_path)
            return
        
        plt.figure(figsize=(10, 8))
        
        # 创建热力图
        try:
            sns.heatmap(head_mask, 
                       cmap=['black', 'grey'],  # 红色表示被kept(0)，绿色表示masked(1)
                       cbar_kws={'label': 'Mask Value (0=kept, 1=masked)'},
                       xticklabels=False, 
                       yticklabels=False)
            
            sparsity_ratio = head_mask.mean()
            plt.title(f'Attention Mask Heatmap\nStage {stage}, Block {block}, {attention_type}, Head {head_idx}\n'
                      f'Sparsity: {sparsity_ratio:.3f} | Shape: {head_mask.shape}')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"保存热力图到: {save_path}")
            
            plt.show()
        except Exception as e:
            print(f"创建热力图失败: {e}")
            # 回退到条形图
            self._plot_1d_mask_as_bar(head_mask.flatten(), stage, block, attention_type, head_idx, save_path)
    
    def _plot_1d_mask_as_bar(self, mask_1d, stage, block, attention_type, head_idx, save_path):
        """将1D mask绘制为条形图"""
        plt.figure(figsize=(12, 6))
        
        # 创建条形图
        indices = np.arange(len(mask_1d))
        colors = ['black' if val == 0 else 'grey' for val in mask_1d]
        
        plt.bar(indices, mask_1d, color=colors, alpha=0.7)
        plt.xlabel('Position Index')
        plt.ylabel('Mask Value (0=masked, 1=kept)')
        plt.title(f'1D Attention Mask\nStage {stage}, Block {block}, {attention_type}, Head {head_idx}\n'
                  f'Length: {len(mask_1d)} | Sparsity: {mask_1d.mean():.3f}')
        plt.grid(True, alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='black', label='Kept (0)'),
            Patch(facecolor='grey', label='Masked (1)')
        ]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        
        if save_path:
            # 修改文件名以指示这是条形图
            bar_path = save_path.replace('.png', '_bar.png')
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            print(f"保存条形图到: {bar_path}")
        
        plt.show()
    
    def visualize_sparsity_trend(self, save_path: Optional[str] = None):
        """可视化稀疏度随stage和block的变化趋势"""
        info_list = self.get_stage_block_info()
        if not info_list:
            return
        
        df = pd.DataFrame(info_list)
        
        # 创建趋势图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 子图1: 按stage的趋势
        stage_avg = df.groupby('stage')['sparsity_ratio'].mean()
        stage_std = df.groupby('stage')['sparsity_ratio'].std()
        
        ax1.errorbar(stage_avg.index, stage_avg.values, yerr=stage_std.values, 
                    marker='o', capsize=5, linewidth=2)
        ax1.set_xlabel('Stage Index')
        ax1.set_ylabel('Average Sparsity Ratio')
        ax1.set_title('Sparsity Trend Across Stages')
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 按attention类型的趋势
        for attn_type in df['attention_type'].unique():
            type_data = df[df['attention_type'] == attn_type]
            type_avg = type_data.groupby('stage')['sparsity_ratio'].mean()
            ax2.plot(type_avg.index, type_avg.values, marker='s', label=attn_type, linewidth=2)
        
        ax2.set_xlabel('Stage Index')
        ax2.set_ylabel('Average Sparsity Ratio')
        ax2.set_title('Sparsity Trend by Attention Type')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"保存趋势图到: {save_path}")
        
        plt.show()
    
    def generate_comprehensive_report(self, output_dir: str = "./sparsity_analysis"):
        """生成综合分析报告"""
        if self.mask_data is None:
            print("没有可分析的数据")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 60)
        print("SPARSITY MASK 综合分析报告")
        print("=" * 60)
        
        # 1. 基本统计分析
        print("\n1. 基本统计分析")
        print("-" * 40)
        
        overall_stats = self.analyze_overall_sparsity()
        if overall_stats:
            df = overall_stats['dataframe']
            print(f"总mask数量: {overall_stats['total_masks']}")
            print(f"平均稀疏度: {overall_stats['avg_sparsity']:.4f} ± {overall_stats['std_sparsity']:.4f}")
            print(f"稀疏度范围: [{overall_stats['min_sparsity']:.4f}, {overall_stats['max_sparsity']:.4f}]")
            
            print("\n按attention类型统计:")
            for attn_type in df['attention_type'].unique():
                type_data = df[df['attention_type'] == attn_type]
                print(f"  {attn_type}: {len(type_data)}个mask, "
                      f"平均稀疏度: {type_data['sparsity_ratio'].mean():.4f}")
        
        # 2. 保存详细数据到CSV
        print("\n2. 保存详细数据到CSV")
        print("-" * 40)
        
        info_list = self.get_stage_block_info()
        if info_list:
            df = pd.DataFrame(info_list)
            csv_path = os.path.join(output_dir, "sparsity_details.csv")
            df.to_csv(csv_path, index=False)
            print(f"详细数据已保存到: {csv_path}")
        
        # 3. 生成可视化
        print("\n3. 生成可视化图表")
        print("-" * 40)
        
        # 趋势图
        trend_path = os.path.join(output_dir, "sparsity_trend.png")
        self.visualize_sparsity_trend(trend_path)
        print(f"趋势图已保存到: {trend_path}")
        
        # 为每个stage的第一个block生成head-wise分析
        stages = df['stage'].unique() if info_list else []
        for stage in stages[1:8]:  # 只分析前3个stage作为示例
            stage_blocks = df[(df['stage'] == stage) & (df['attention_type'] == 'self_attention')]['block']
            if len(stage_blocks) > 0:
                block = min(stage_blocks)
                
                # head-wise稀疏度分析
                head_path = os.path.join(output_dir, f"stage_{stage}_block_{block}_head_sparsity.png")
                self.visualize_head_sparsity(stage, block, 'self_attention', head_path)
                print(f"Head-wise稀疏度分析图已保存到: {head_path}")
                
                # 检查mask形状，如果是1D或3D则使用条形图
                key = f"stage_{stage}_block_{block}"
                if (key in self.mask_data['masks'] and 
                    'self_attention' in self.mask_data['masks'][key]):
                    mask_tensor = self.mask_data['masks'][key]['self_attention']['mask']
                    if mask_tensor.dim() == 3 and mask_tensor.shape[1] == 1 and mask_tensor.shape[2] == 1:
                        # 使用条形图
                        heatmap_path = os.path.join(output_dir, f"stage_{stage}_block_{block}_bar.png")
                        self._plot_1d_mask_as_bar(
                            mask_tensor[0].flatten().numpy(), 
                            stage, block, 'self_attention', 0, heatmap_path
                        )
                        print(f"Mask条形图已保存到: {heatmap_path}")
                    else:
                        # 使用热力图
                        heatmap_path = os.path.join(output_dir, f"stage_{stage}_block_{block}_heatmap.png")
                        self.visualize_mask_heatmap(stage, block, 'self_attention', 0, heatmap_path)
                        print(f"Mask热力图已保存到: {heatmap_path}")
        
        # 4. 生成统计摘要
        print("\n4. 统计摘要")
        print("-" * 40)
        
        summary_path = os.path.join(output_dir, "analysis_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Sparsity Mask Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"数据文件: {self.pkl_path}\n")
            f.write(f"分析时间: {pd.Timestamp.now()}\n\n")
            
            if overall_stats:
                f.write("总体统计:\n")
                f.write(f"  总mask数量: {overall_stats['total_masks']}\n")
                f.write(f"  平均稀疏度: {overall_stats['avg_sparsity']:.4f}\n")
                f.write(f"  稀疏度标准差: {overall_stats['std_sparsity']:.4f}\n")
                f.write(f"  最小稀疏度: {overall_stats['min_sparsity']:.4f}\n")
                f.write(f"  最大稀疏度: {overall_stats['max_sparsity']:.4f}\n\n")
            
            f.write("Stage统计:\n")
            for stage in stages:
                stage_data = df[df['stage'] == stage]
                f.write(f"  Stage {stage}: {len(stage_data)}个mask, "
                       f"平均稀疏度: {stage_data['sparsity_ratio'].mean():.4f}\n")
        
        print(f"分析摘要已保存到: {summary_path}")
        print(f"\n完整分析报告已生成到目录: {output_dir}")
        print("=" * 60)

# 使用示例
def main():
    # 替换为您的pkl文件路径
    pkl_file = "./sparsity_masks/mask_data.pkl"  # 修改为实际路径
    
    # 创建分析器
    analyzer = SparsityMaskAnalyzer(pkl_file)
    
    if analyzer.mask_data is None:
        print("无法加载数据，请检查文件路径")
        return
    
    # 方法1: 生成完整报告
    analyzer.generate_comprehensive_report("./sparsity_analysis_report")
    
    # 方法2: 单独分析特定stage和block
    print("\n单独分析示例:")
    
    # 分析stage 0, block 0的self attention
    results = analyzer.analyze_head_sparsity(0, 0, 'self_attention')
    if results:
        print(f"Stage 0, Block 0, Self Attention:")
        print(f"  Head数量: {results['num_heads']}")
        print(f"  平均稀疏度: {results['avg_head_sparsity']:.4f}")
        print(f"  稀疏度范围: [{results['min_head_sparsity']:.4f}, {results['max_head_sparsity']:.4f}]")
    
    # 可视化特定mask
    analyzer.visualize_mask_heatmap(0, 0, 'self_attention', 0, 
                                   "./sparsity_analysis_report/sample_heatmap.png")

        
def compare_multiple_experiments(pkl_files: List[str], experiment_names: List[str], 
                               output_dir: str = "./comparison_analysis"):
    """
    比较多个实验的sparsity mask结果
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    for pkl_file, exp_name in zip(pkl_files, experiment_names):
        analyzer = SparsityMaskAnalyzer(pkl_file)
        if analyzer.mask_data is not None:
            info_list = analyzer.get_stage_block_info()
            if info_list:
                df = pd.DataFrame(info_list)
                df['experiment'] = exp_name
                all_results.append(df)
    
    if not all_results:
        print("没有有效数据可比较")
        return
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # 创建比较图
    plt.figure(figsize=(12, 6))
    
    for exp_name in combined_df['experiment'].unique():
        exp_data = combined_df[combined_df['experiment'] == exp_name]
        stage_avg = exp_data.groupby('stage')['sparsity_ratio'].mean()
        plt.plot(stage_avg.index, stage_avg.values, marker='o', label=exp_name, linewidth=2)
    
    plt.xlabel('Stage Index')
    plt.ylabel('Average Sparsity Ratio')
    plt.title('Sparsity Comparison Across Experiments')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    comparison_path = os.path.join(output_dir, "experiment_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"实验比较图已保存到: {comparison_path}")
    
    # 保存比较数据
    csv_path = os.path.join(output_dir, "comparison_data.csv")
    combined_df.to_csv(csv_path, index=False)
    print(f"比较数据已保存到: {csv_path}")

def analyze_mask_patterns(analyzer: SparsityMaskAnalyzer, stage: int, block: int, 
                        attention_type: str, num_heads: int = 4):
    """
    分析mask的模式（如对角线模式、块状模式等）
    """
    key = f"stage_{stage}_block_{block}"
    
    if (analyzer.mask_data is None or 
        key not in analyzer.mask_data['masks'] or 
        attention_type not in analyzer.mask_data['masks'][key]):
        return
    
    mask_data = analyzer.mask_data['masks'][key][attention_type]
    mask_tensor = mask_data['mask']
    num_heads_total = mask_tensor.shape[0]
    
    # 分析前几个head的模式
    fig, axes = plt.subplots(2, min(2, num_heads//2), figsize=(15, 8))
    axes = axes.flatten()
    
    for i, head_idx in enumerate(range(min(num_heads, len(axes)))):
        head_mask = mask_tensor[head_idx].numpy()
        
        # 计算各种模式指标
        diagonal_strength = np.mean(np.diag(head_mask)) if head_mask.shape[0] == head_mask.shape[1] else 0
        blockiness = analyze_block_pattern(head_mask)
        
        # 可视化
        im = axes[i].imshow(head_mask, cmap='viridis')
        axes[i].set_title(f'Head {head_idx}\nDiag: {diagonal_strength:.3f}, Block: {blockiness:.3f}')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()

def analyze_block_pattern(mask: np.ndarray, block_size: int = 8) -> float:
    """
    分析mask的块状模式强度
    """
    h, w = mask.shape
    blockiness = 0
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = mask[i:i+block_size, j:j+block_size]
            if block.size > 0:
                # 计算块内一致性
                block_std = np.std(block)
                blockiness += 1 - block_std  # 标准差越小，块状模式越强
    
    return blockiness / ((h//block_size) * (w//block_size)) if h//block_size > 0 and w//block_size > 0 else 0

# 使用示例

if __name__ == "__main__":
    main()