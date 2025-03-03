import json
from tqdm import tqdm
import os
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

data = json.load(open("/raid_sdd/zzy/data/halle/llava_v1_5_mix665k.json","r"))
image_path = "/raid_sdd/zzy/data/halle"


def check_image(dd):
    """检查图片是否可以正常打开"""
    if 'image' in dd:
        image_file = os.path.join(image_path, dd['image'])
        try:
            # 尝试打开图片
            with Image.open(image_file).convert('RGB') as img:
                pass  # 如果可以正常打开，不做任何处理
        except Exception as e:
            # 返回无法正常读取的图片及错误信息
            return (dd['image'], str(e))
    return None

# 保存所有出错的图片
corrupted_images = []

# 使用多线程来检查图片读取
with ThreadPoolExecutor(max_workers=64) as executor:
    # tqdm 用来显示多线程的任务进度
    for result in tqdm(executor.map(check_image, data), total=len(data)):
        if result is not None:  # 有异常的图片
            corrupted_images.append(result)

# 输出无法读取的图片
if corrupted_images:
    print("以下图片无法正常读取：")
    for image, error in corrupted_images:
        print(f"{image}: {error}")
else:
    print("所有图片都可以正常读取。")