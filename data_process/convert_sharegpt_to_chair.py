import json

input_file = "/home/zangzeyuan/data/halle/detail_sample_9k.json"
output_file = "/home/zangzeyuan/data/halle/detail_sample_9k.jsonl"
base_path = ""

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_file, "w", encoding="utf-8") as fout:
    for item in data:
        image_id = int(item["id"])
        image_file = base_path + item["image"]
        caption = next(
            conv["value"] for conv in item["conversations"] if conv["from"] == "gpt"
        )

        out = {
            "image_id": image_id,
            "image_file": image_file,
            "caption": caption
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")