import torch
import pandas as pd
from PIL import Image
from openpyxl.styles.builtins import output
from pygments.lexer import default
import transformers, peft, accelerate
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
import argparse
import re
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
from ..utils.evaluation import eval_classification, save_classification_report, extract_think_answer, extract_prediction_rationale


def get_response(text, image_path, model, processor):
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": text},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text


def judge_label(text):
    text = text.lower()
    if "yes" in text:
        return 1
    elif "no" in text:
        return 0



def round_result(result: float):
    """将结果转换为百分比得分形式，同时保留两位小数"""
    return round(result * 100, 2)


def reade_file(file_path: str):
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


def save_data(image_data: list[dict], save_path: str):
    """
    传入推理数据并保存
    """
    if os.path.exists(save_path):
        output = pd.read_excel(save_path)
        df = pd.DataFrame(image_data)
        output = output._append(df)
        output.to_excel(save_path, index=False)
    else:
        # 创建一下父目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df = pd.DataFrame(image_data)
        df.to_excel(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=r"")
    parser.add_argument("--dataset_name", default="hmc")
    parser.add_argument("--image_dir", default="")
    parser.add_argument("--output_dir",default=r"")
    parser.add_argument("--output_name", default=r"")
    parser.add_argument("--eval_save_dir", default=r"")
    parser.add_argument("--query",
                        default="<image>Is it hateful? The answer should be Yes or No, if the meme is hateful than output Yes ; if the meme is Benign than output No.",
                        help='Query to pass to the model')

    # 模型路径
    parser.add_argument("--base_model_path", default="")
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--comment", type=str, default=None)
    args = parser.parse_args()

    base_model_path = args.base_model_path
    lora_path = args.lora_path

    # 加载模型, 适用lora
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(base_model_path)

    print("transformers =", transformers.__version__)
    print("peft =", peft.__version__)
    print("accelerate =", accelerate.__version__)
    print("hf_device_map =", getattr(base_model, "hf_device_map", None))
    print("base_model._no_split_modules =", getattr(base_model, "_no_split_modules", None))
    print("base_model.model._no_split_modules =",
          getattr(getattr(base_model, "model", None), "_no_split_modules", None))


    if lora_path:
        print("采用lora模型测评...")
        model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        print("采用基础模型测评...")
        model = base_model
    model.eval()

    # 读取测试集
    dataset_dir = args.dataset_dir
    data_set = reade_file(dataset_dir)
    output_path = os.path.join(args.output_dir, args.output_name)

    # 保存之前结果用于后续测试数据是否被评估过
    before_data = []
    if os.path.exists(output_path):
        before_output = pd.read_excel(output_path)
        before_data = before_output['image_name'].tolist()

    # 2分类
    # hmc 2-class: "id":42953, "img":"img\/42953.png", "label":0, "text":"its their character not their color that matters"
    # mami 2/mutil-class:   file_name	    misogynous  shaming	    stereotype      objectification     violence    text
    #                       1.jpg	            0	        0	         0	            0	                0	    Milk Milk.zip
    # pridemm 2/mutil-class: file_name	    hate	    target	    stance	    humour
    #                        img_1.png	      0	         NAN	      0	          1
    # toxicn-mm 2/mutil-class:  name	        label	    type    text
    #                           1936.jpg		  0	         0      妈妈：如果你朋友从悬崖上 跳下来，你也会跳吗！ 我：我会先跳！ 我媽:
    target_names = ["not-hateful", "hateful"]

    # 根据不同的数据集进行不同的处理, 得到每个数据集的图像路径和标签名字
    if "hmc" in args.dataset_name:
        image_label = "img"
        target_label = "label"
    elif args.dataset_name.lower() == "toxicn-mm":
        image_label = "name"
        target_label = "label"
    elif args.dataset_name.lower() == "mami":
        image_label = "file_name"
        target_label = "misogynous"
    elif args.dataset_name.lower() == "pridemm":
        image_label = "file_name"
        target_label = "hate"
    else:
        raise ValueError(
            f"Unsupported dataset name: {args.dataset_name}")

    # 模型推理, 记录推理结果, 保存推理结果
    off_gt = []
    off_pred = []
    data = []
    for index, row in tqdm(data_set.iterrows(), total=len(data_set)):
        image_name = row[image_label]
        if  "hmc" in args.dataset_name:
            image_name = image_name.split("/")[-1]
        image_path = os.path.join(args.image_dir, image_name)

        # 跳过已经处理过的图片, 避免重复处理, 同时提取当前图片的推理结果标签值
        if image_name in before_data:
            row = before_output[before_output["image_name"] == image_name].iloc[0]
            y_true = row["gold_label"]
            y_pred = row["pred_label"]
            off_gt.append(y_true)
            off_pred.append(y_pred)
            continue

        gold_label = int(row[target_label])

        question = args.query
        response = get_response(question, image_path, model, processor)


        if type(response) == list:
            response = response[0]

        # 如果是 think + answer 模式，提取 answer
        if "think" in args.dataset_name:
            response_dict = extract_think_answer(response)
            response = response_dict["answer"]
        # 如果是 explain 模式 (先预测, 后解释)，提取 prediction
        elif "explain" in args.dataset_name:
            response_dict = extract_prediction_rationale(response)
            response = response_dict["prediction"]

        pred_label = judge_label(response)

        off_gt.append(gold_label)
        off_pred.append(pred_label)

        data.append(
            {
                "image_name": image_name,
                "gold_label": gold_label,
                "pred_label": pred_label,
                "response": response,
                "comment": args.comment,
            }
        )
        if index != 0 and index % 50 == 0:
            save_data(data, output_path)
            print(f"当前处理到第{index}张图片, 保存一下之前的数据")
            print()
            data = []


    save_data(data, output_path)
    print(f"数据推理完毕, 推理结果保存在{output_path}中")
    print()

    report, confusion = eval_classification(off_gt, off_pred, target_names)
    save_classification_report(off_gt, off_pred, target_names, args.eval_save_dir, args.output_name.split(".")[0],
                               args.comment)
