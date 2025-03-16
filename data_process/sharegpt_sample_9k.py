import json
import random

data_path = "/raid_sdd/zzy/data/halle/sharegpt4v_instruct_gpt4-vision_part_coco_50k.json"
# data_path = "/raid_sdd/whz/data/halle/detail_switch_100_part.json"
random.seed(42)

with open(data_path, "r") as f:
    tmp_data = json.load(f)

num_samples = 9093
if len(tmp_data) < num_samples:
    raise ValueError("not enough data")

sampled_data = random.sample(tmp_data, num_samples)

output_path = "/raid_sdd/zzy/data/halle/sharegpt4v_instruct_gpt4-vision_part_coco_sample_9k.json"
with open(output_path, "w") as f:
    json.dump(sampled_data, f, indent=4)


print(f"成功取样 {num_samples} 条数据，并保存至 {output_path}")