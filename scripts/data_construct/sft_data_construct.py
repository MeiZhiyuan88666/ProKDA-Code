import pandas as pd
import argparse
import json
import math
import os

import datasets
from PIL import Image
from tqdm import tqdm

def save_json(json_data, json_path):
    # 将json数据保存到本地
    if not os.path.exists(os.path.dirname(json_path)):
        os.makedirs(os.path.dirname(json_path))

    with open(json_path, 'w', encoding="utf-8") as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)
    print(f'数据已保存到 {json_path}')


def reade_file(file_path):
    """
    读取文件并返回
    """
    data_type = file_path.split(".")[-1]
    if data_type == "jsonl":
        dataset = pd.read_json(file_path, lines=True)
    elif data_type == "json":
        dataset = pd.read_json(file_path)
    elif data_type == "xlsx":
        dataset = pd.read_excel(file_path)
    else:
        raise ValueError("Invalid data type")
    return dataset

if __name__ == '__main__':



    args_default_dataset_dir = r""
    args_default_dataset_name = ""
    args_default_image_dir = ""
    args_default_output_dir = r""
    args_default_output_name = ""
    args_default_comment = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=args_default_dataset_dir)
    parser.add_argument("--dataset_name", default=args_default_dataset_name)
    parser.add_argument("--output_name", default=args_default_output_name)
    parser.add_argument("--image_dir", default=args_default_image_dir)
    parser.add_argument("--output_dir", default=args_default_output_dir)
    parser.add_argument("--comment", type=str, default=None)
    args = parser.parse_args()

    # Load data
    dataset_dir = args.dataset_dir
    data_set = reade_file(dataset_dir)


    # Create question and answer  2分类
    # hmc 2-class: "id":42953, "img":"img\/42953.png", "label":0, "text":"its their character not their color that matters"
    # mami 2/mutil-class:   file_name	    misogynous  shaming	    stereotype      objectification     violence    text
    #                       1.jpg	            0	        0	         0	            0	                0	    Milk Milk.zip
    # pridemm 2/mutil-class: file_name	    hate	    target	    stance	    humour
    #                        img_1.png	      0	         NAN	      0	          1
    # toxicn-mm 2/mutil-class:  name	        label	    type    text
    #                           1936.jpg		  0	         0      妈妈：如果你朋友从悬崖上 跳下来，你也会跳吗！ 我：我会先跳！ 我媽:
    if args.dataset_name == "hmc":
        question = "<image>Is it hateful?"
        image_label = "img"
        target_label = "label"
    elif args.dataset_name.lower() == "toxicn-mm":
        question = "<image>Is it harmful?"
        image_label = "name"
        target_label = "label"
    elif args.dataset_name.lower() == "mami":
        question = "<image>Is it a misogynistic meme?"
        image_label = "file_name"
        target_label = "misogynous"
    elif args.dataset_name.lower() == "pridemm":
        question = "<image>Is it a hateful meme towards LGBTQ+?"
        image_label = "file_name"
        target_label = "hate"
    else:
        raise ValueError(
            f"Unsupported dataset name: {args.dataset_name}")


    data = []

    for index, row in tqdm(data_set.iterrows(), total=len(data_set)):
        image_name = row[image_label]
        if args.dataset_name == "hmc":
            image_name = image_name.split("/")[-1]
        image_path = os.path.join(args.image_dir, image_name)

        label = row[target_label]

        if args.dataset_name == "hmc":
            answer = "Yes. It is a hateful meme." if label == 1 else "No. It is not a hateful meme."
        elif args.dataset_name.lower() == "toxicn-mm":
            answer = "Yes. It is a harmful meme." if label == 1 else "No. It is not a harmful meme."
        elif args.dataset_name.lower() == "mami":
            answer = "Yes. It is a misogynistic meme." if label == 1 else "No. It is not a misogynistic meme."
        elif args.dataset_name.lower() == "pridemm":
            answer = "Yes. It is a hateful meme." if label == 1 else "No. It is not a hateful meme."

        data.append({
            "id": index,
            "messages": [
                {
                    "content": question,
                    "role": "user"
                },
                {
                    "content": answer,
                    "role": "assistant"
                }
            ],
            "images": [
                image_path
            ],
            "label": label
        })

    save_path = os.path.join(args.output_dir, args.output_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_json(data, save_path)
    print(f"数据已保存到 {save_path}")




























