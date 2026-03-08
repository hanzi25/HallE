import json
import random

data_path = "/home/zangzeyuan/data/halle/flickr30k_karpathy_train_conversation_single_sample.json"
# data_path = "/home/zangzeyuan/data/halle/detail_23k.json"
# data_path = "/raid_sdd/whz/data/halle/detail_switch_100_part.json"
random.seed(42)

with open(data_path, "r") as f:
    tmp_data = json.load(f)

num_samples = 1000
# num_samples = 9093
if len(tmp_data) < num_samples:
    raise ValueError("not enough data")

sampled_data = random.sample(tmp_data, num_samples)

output_path = "/home/zangzeyuan/data/halle/flickr30k_karpathy_train_conversation_single_1k.json"
with open(output_path, "w") as f:
    json.dump(sampled_data, f, indent=4)


print(f"成功取样 {num_samples} 条数据，并保存至 {output_path}")