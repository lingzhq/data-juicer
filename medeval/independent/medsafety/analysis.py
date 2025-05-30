import json
from collections import Counter
import matplotlib.pyplot as plt

def load_results(file_path):
    scores = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if data['status'] == 'success' and data['score'] is not None:
                scores.append(data['score'])
    return scores

def plot_score_distribution(scores, save_path='./medsafety/data/score_distribution.png'):
    score_counts = Counter(scores)
    
    for score in range(1, 6):
        if score not in score_counts:
            score_counts[score] = 0
    
    sorted_scores = sorted(score_counts.items())
    x = [str(score) for score, _ in sorted_scores]
    counts = [count for _, count in sorted_scores]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, counts, width=0.8, color='#1f77b4')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')
    
    plt.title('Model Safety Score Distribution', fontsize=14)
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(axis='y', alpha=0.4)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"图表已保存至：{save_path}")
    plt.show()

if __name__ == "__main__":
    scores = load_results('./medsafety/data/res.jsonl')
    
    print(f"总样本数：{len(scores)}")
    print("分数分布：")
    for score, count in sorted(Counter(scores).items()):
        print(f"Score {score}: {count} 条")
    
    plot_score_distribution(scores)