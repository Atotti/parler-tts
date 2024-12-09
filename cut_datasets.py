from datasets import load_dataset

# 設定
descriptions_dataset_name = "Atotti/tomoko-tts-descriptions-v1"  # 元のデータセット名
source_dataset_name = "Atotti/jsut-corpus-datasets"  # 元のデータセット名

subset_name = "default"  # サブセット名がある場合指定
n = 1250  # 抽出する件数
new_descriptions_dataset_name = f"{descriptions_dataset_name}-{n}"  # 新しいデータセット名
new_dataset_name = f"{source_dataset_name}-{n}"  # 新しいデータセット名

# データセットの読み込み
if subset_name:
    description_dataset = load_dataset(descriptions_dataset_name, subset_name)
    dataset = load_dataset(source_dataset_name, subset_name)
else:
    description_dataset = load_dataset(descriptions_dataset_name)
    dataset = load_dataset(source_dataset_name)

# n件だけ取得
def select_n_examples(dataset_dict, n):
    """DatasetDictの各分割からn件を取得"""
    return {split: dataset_dict[split].select(range(min(n, len(dataset_dict[split])))) for split in dataset_dict.keys()}

subset_description_data = select_n_examples(description_dataset, n)
subset_data = select_n_examples(dataset, n)

# データセットをHugging Face Hubにプッシュ
subset_description_data = {split: subset_description_data[split] for split in subset_description_data}
subset_data = {split: subset_data[split] for split in subset_data}

for split, data in subset_description_data.items():
    data.push_to_hub(f"{new_descriptions_dataset_name}-{split}", private=True)

for split, data in subset_data.items():
    data.push_to_hub(f"{new_dataset_name}-{split}", private=True)

print("データセットが正常にアップロードされました。")
