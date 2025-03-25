import json
import random
import numpy as np

caption_data_path = "/raid_sdd/zzy/data/halle/sharegpt4v_instruct_gpt4-vision_part_coco_50k.json"
qa_data_path = "/raid_sdd/zzy/data/halle/conversation_58k.json"
reasoning_data_path = "/raid_sdd/zzy/data/halle/complex_reasoning_77k.json"
# data_path = "/raid_sdd/whz/data/halle/detail_switch_100_part.json"
random.seed(42)

joint_data = []

#### Caption
with open(caption_data_path, "r") as f:
    tmp_data = json.load(f)

num_caption_samples = 4000
if len(tmp_data) < num_caption_samples:
    raise ValueError("not enough data")

sampled_data = random.sample(tmp_data, num_caption_samples)

joint_data.extend(sampled_data)


#### QA
with open(qa_data_path, "r") as f:
    tmp_data = json.load(f)

num_qa_samples = 4000
if len(tmp_data) < num_qa_samples:
    raise ValueError("not enough data")

sampled_data = random.sample(tmp_data, num_qa_samples)

joint_data.extend(sampled_data)


#### Reasoning
with open(reasoning_data_path, "r") as f:
    tmp_data = json.load(f)

num_reasoning_samples = 2000
if len(tmp_data) < num_reasoning_samples:
    raise ValueError("not enough data")

sampled_data = random.sample(tmp_data, num_reasoning_samples)

joint_data.extend(sampled_data)


output_path = f"/raid_sdd/zzy/data/halle/joint_caption_{num_caption_samples/1000}k_qa_{num_qa_samples/1000}k_reasoning_{num_reasoning_samples/1000}k.json"

with open(output_path, "w") as f:
    json.dump(joint_data, f, indent=4)


print(f"成功取样 {(num_caption_samples+num_qa_samples+num_reasoning_samples)//1000}k 条数据，并保存至 {output_path}")