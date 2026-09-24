import pandas as pd
import argparse
import json
import math
import os
from collections import Counter
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


if __name__ == '__main__':


    args_default_dataset_dir = r""
    args_default_dataset_name = "sft_hmc"
    args_default_image_dir = ""
    args_default_output_dir = r""
    args_default_output_name = ""
    args_default_confidence_threshold = 60      # 置信度小于 xx%的样本
    args_default_comment = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=args_default_dataset_dir)
    parser.add_argument("--dataset_name", default=args_default_dataset_name)
    parser.add_argument("--image_dir", default=args_default_image_dir)
    parser.add_argument("--output_dir", default=args_default_output_dir)
    parser.add_argument("--output_name", default=args_default_output_name)
    parser.add_argument("--confidence_threshold", default=args_default_confidence_threshold)
    parser.add_argument("--comment", type=str, default=args_default_comment)
    args = parser.parse_args()

    # Load data
    dataset_dir = args.dataset_dir
    data_type = dataset_dir.split(".")[-1]
    if data_type == "jsonl":
        data_set = pd.read_json(dataset_dir, lines=True)
    elif data_type == "json":
        data_set = pd.read_json(dataset_dir)
    elif data_type == "xlsx":
        data_set = pd.read_excel(dataset_dir)

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
    elif args.dataset_name.lower() == "sft_hmc":    # sft后对训练集进行推理, 将预测失败的样本做成dpo
        question = "<image>Is it hateful?"
        image_label = "image_name"
        target_label = "gold_label"
        pred_label = "pred_label"
        confidence = "confidence"
        # image_name,gold_label,pred_label,response
    else:
        raise ValueError(
            f"Unsupported dataset name: {args.dataset_name}")


    data = []   # 存储数据
    num_counts = 0  # 样本数量
    label_counter = Counter()   # 统计标签数量
    confidence_threshold = args.confidence_threshold    # 置信度小于90%的样本
    for index, row in tqdm(data_set.iterrows(), total=len(data_set)):
        image_name = row[image_label]
        if args.dataset_name == "hmc":
            image_name = image_name.split("/")[-1]
        image_path = os.path.join(args.image_dir, image_name)

        gold_label = row[target_label]

        # 这时候要根据真实标签来生成一对chosen 和 rejected
        if args.dataset_name == "hmc" or args.dataset_name == "sft_hmc":
            yes_answer = "Yes. It is a hateful meme."
            no_answer = "No. It is not a hateful meme."
            chosen_answer = yes_answer if gold_label == 1 else no_answer
            rejected_answer = no_answer if gold_label == 1 else yes_answer

        elif args.dataset_name.lower() == "toxicn-mm" or args.dataset_name == "sft_toxicn-mm":
            yes_answer = "Yes. It is a harmful meme."
            no_answer = "No. It is not a harmful meme."
            chosen_answer = yes_answer if gold_label == 1 else no_answer
            rejected_answer = no_answer if gold_label == 1 else yes_answer

        elif args.dataset_name.lower() == "mami" or args.dataset_name == "sft_mami":
            yes_answer = "Yes. It is a misogynistic meme."
            no_answer = "No. It is not a misogynistic meme."
            chosen_answer = yes_answer if gold_label == 1 else no_answer
            rejected_answer = no_answer if gold_label == 1 else yes_answer

        elif args.dataset_name.lower() == "pridemm" or args.dataset_name == "sft_pridemm":
            yes_answer = "Yes. It is a hateful meme."
            no_answer = "No. It is not a hateful meme."
            chosen_answer = yes_answer if gold_label == 1 else no_answer
            rejected_answer = no_answer if gold_label == 1 else yes_answer
        else:
            raise ValueError(
                f"Unsupported dataset name: {args.dataset_name}")

        # 如果是sft数据集, 则需要判断置信度是否<90%(置信度门槛), 如果大于门槛并且回答正确则跳过
        if "sft" in args.dataset_name:
            try:
                pre_label_ = row[pred_label]
                if pre_label_ == gold_label and row[confidence] >= args.confidence_threshold:
                    continue
            except:
                print(f"{args.dataset_name} has no pred_label")

        label_counter[gold_label] += 1

        data.append({
            "id": index,
            "messages": [
                {
                    "content": question,
                    "role": "user"
                }
            ],
            "chosen": {
                "role": "assistant",
                "content": chosen_answer
            },
            "rejected": {
                "role": "assistant",
                "content": rejected_answer
            },
            "images": [
                image_path
            ],
            "label": gold_label
        })
        num_counts += 1

    save_path = os.path.join(args.output_dir, args.output_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_json(data, save_path)
    for label, count in label_counter.items():
        print(f"Label {label}: {count} samples")
    print(f"Saved {num_counts} samples to {save_path}")





























