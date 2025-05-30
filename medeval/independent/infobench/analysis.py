import json

def calculate_accuracy(stats):
    total = stats['true'] + stats['false']
    return (stats['true'] / total) * 100 if total > 0 else 0.0

def main():
    # 初始化统计字典
    stats = {
        'Easy_set': {'entries': 0, 'true': 0, 'false': 0},
        'Hard_set': {'entries': 0, 'true': 0, 'false': 0},
        'overall': {'entries': 0, 'true': 0, 'false': 0}
    }

    try:
        with open('./infobench/data/qwen3-32b/infobench_DecomposeEval.json', 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                subset = entry['subset']
                eval_list = entry['eval']

                # 更新各子集统计
                if subset in ['Easy_set', 'Hard_set']:
                    stats[subset]['entries'] += 1
                    for value in eval_list:
                        if value:
                            stats[subset]['true'] += 1
                        else:
                            stats[subset]['false'] += 1

                # 更新全局统计
                stats['overall']['entries'] += 1
                for value in eval_list:
                    if value:
                        stats['overall']['true'] += 1
                    else:
                        stats['overall']['false'] += 1

        # 打印结果
        for subset in ['Easy_set', 'Hard_set', 'overall']:
            data = stats[subset]
            print(f"=== {subset} ===")
            print(f"数据条目数: {data['entries']}")
            print(f"总评估项数: {data['true'] + data['false']}")
            print(f"True数量: {data['true']}")
            print(f"False数量: {data['false']}")
            print(f"正确率: {calculate_accuracy(data):.2f}%\n")

    except FileNotFoundError:
        print("错误：文件未找到，请检查文件路径")
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()
