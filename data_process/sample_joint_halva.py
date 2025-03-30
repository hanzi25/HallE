import json
import os
import random
from collections import Counter

caption_data_path = "/raid_sdd/zzy/data/halle/sharegpt4v_instruct_gpt4-vision_part_coco_50k.json"
qa_data_path = "/raid_sdd/zzy/data/halle/conversation_58k.json"
reasoning_data_path = "/raid_sdd/zzy/data/halle/complex_reasoning_77k.json"
halva_data_path = "/raid_sdd/zzy/data/halle/halva_instruct_21k.json"
coco_path = "coco/train2017"

random.seed(42)

joint_data = []
num_caption_samples = 5000
num_qa_samples = 3000
num_reasoning_samples = 1000
num_halva_samples = 1000

#### Caption
with open(caption_data_path, "r") as f:
    tmp_data = json.load(f)
new_tmp_data = []
for item in tmp_data:
    new_item = item
    new_item["image"] = os.path.join(coco_path, item["image"])
    new_tmp_data.append(new_item)
if len(new_tmp_data) < num_caption_samples:
    raise ValueError("not enough data")
sampled_data = random.sample(new_tmp_data, num_caption_samples)
joint_data.extend(sampled_data)

#### QA
with open(qa_data_path, "r") as f:
    tmp_data = json.load(f)
new_tmp_data = []
for item in tmp_data:
    new_item = item
    new_item["image"] = os.path.join(coco_path, item["image"])
    new_tmp_data.append(new_item)
if len(new_tmp_data) < num_qa_samples:
    raise ValueError("not enough data")
sampled_data = random.sample(new_tmp_data, num_qa_samples)
joint_data.extend(sampled_data)

#### Reasoning
with open(reasoning_data_path, "r") as f:
    tmp_data = json.load(f)
new_tmp_data = []
for item in tmp_data:
    new_item = item
    new_item["image"] = os.path.join(coco_path, item["image"])
    new_tmp_data.append(new_item)
if len(new_tmp_data) < num_reasoning_samples:
    raise ValueError("not enough data")
sampled_data = random.sample(new_tmp_data, num_reasoning_samples)
joint_data.extend(sampled_data)

#### Halva
with open(halva_data_path, "r") as f:
    tmp_data = json.load(f)
if len(tmp_data) < num_halva_samples:
    raise ValueError("not enough data")
type_counts = Counter(item['type'] for item in tmp_data)
num_halva_qa_samples = num_halva_samples * type_counts["qa"] // (type_counts["qa"] + type_counts["detailed cap"])
num_halva_cap_samples = num_halva_samples * type_counts["detailed cap"] // (type_counts["qa"] + type_counts["detailed cap"])
halva_qa_data = [item for item in tmp_data if item["type"] == "qa"]
halva_cap_data = [item for item in tmp_data if item["type"] == "detailed cap"]
sampled_qa_data = random.sample(halva_qa_data, num_halva_qa_samples)
sampled_cap_data = random.sample(halva_cap_data, num_halva_cap_samples)
joint_data.extend(sampled_qa_data)
joint_data.extend(sampled_cap_data)

output_path = f"/raid_sdd/zzy/data/halle/joint_caption_{num_caption_samples / 1000}k_qa_{num_qa_samples / 1000}k_reasoning_{num_reasoning_samples / 1000}k_halva_detailed+qa_{num_halva_samples / 1000}k.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(joint_data, f, indent=4, ensure_ascii=False)

print(f"Conversion completed, output: {output_path}")