import json
from tqdm import tqdm

data_path = "/raid_sdd/whz/data/halle/detail_switch_23k.json"
# data_path = "/raid_sdd/whz/data/halle/detail_switch_100_part.json"

tmp_data = json.load(open(data_path, "r"))

data = list()
for dd in tqdm(tmp_data):
    if dd['hall_factor'] == -1:
        data.append(dd)

with open("/raid_sdd/whz/data/halle/detail_switch_minus_1_9093.json", "w") as f:
    f.write(json.dumps(data))