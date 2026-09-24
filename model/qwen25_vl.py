import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
import argparse
import numpy as np
from tqdm import tqdm
import os


# 运行 qwen2.5vl 模型
# 可以获取 预测标签token的概率

chat_text_template = (
)


def get_response(text, image_path, model, processor):
    """
    简单的模型对话, 输入文本、图像路径、模型以及处理器，返回输出文本
    Args:
        text:
        image_path:
        model:
        processor:

    Returns:

    """
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




def get_binary_probs_qwen_vl(model, processor: AutoProcessor, image_path: str, chat_text: str, token_pos_style="plain"):
    """
    用来测试模型判断meme是否仇恨, 返回仇恨和非仇恨概率以及当前输出文本. 要求模型只输出1/0来表示仇恨和非仇恨
    Args:
        model: 传入模型
        processor:  处理器
        image_path: 图像路径
        meme_text:  meme的ocr文本
        token_pos_style: token的位置风格
    """
    image = Image.open(image_path).convert("RGB")
    tokenizer = processor.tokenizer
    model.eval()

    # 1) 根据 tokenizer 选择合适的标签token
    if token_pos_style == "plain":
        token_1 = "1"
        token_0 = "0"
    elif token_pos_style == "space":
        token_1 = " 1"
        token_0 = " 0"
    else:
        raise ValueError("token_pos_style must be 'plain' or 'space'")

    ids_1 = tokenizer.encode(token_1, add_special_tokens=False)
    ids_0 = tokenizer.encode(token_0, add_special_tokens=False)

    if len(ids_1) != 1 or len(ids_0) != 1:
        raise ValueError(
            f"Chosen label tokens are not single-token. "
            f"{repr(token_1)} -> {ids_1}, {repr(token_0)} -> {ids_0}"
        )

    id_1 = ids_1[0]
    id_0 = ids_0[0]


    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": chat_text
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
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

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512,
                                 return_dict_in_generate=True,
                                 output_logits=True)

    output_ids = outputs.sequences
    logits = outputs.logits

    next_token_logits = logits[0][0, :]
    logit_1 = next_token_logits[id_1]
    logit_0 = next_token_logits[id_0]
    binary_logits = torch.stack([logit_1, logit_0], dim=-1)
    binary_probs = F.softmax(binary_logits, dim=-1)
    p_harmful = binary_probs[0].item()
    p_benign = binary_probs[1].item()
    pred = 1 if p_harmful >= p_benign else 0
    confidence = max(p_harmful, p_benign)

    generated_ids = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids.repeat(output_ids.shape[0], 1), output_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # print("output_text: ", output_text)
    return {
        "logit_0": logit_0.item(),
        "logit_1": logit_1.item(),
        "p_harmful": p_harmful,
        "p_benign": p_benign,
        "pred_label": pred,
        "confidence": confidence,
        "token_1": token_1,
        "token_0": token_0,
        "id_1": id_1,
        "id_0": id_0,
        "output_text": output_text[0],
    }



def get_batch_response(texts, image_paths, model, processor):
    """
    批量推理数据
    Args:
        texts: list[str]
        image_paths: list[str]
        model:
        processor:
    Returns: list[str]
    """
    # 要求text和image_path的数量一致的, 要不然分不清
    if len(texts) != len(image_paths):
        raise ValueError("texts and images must have the same length")

    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    messages = []

    for text, image in zip(texts, images):
        messages.append([
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
        ])

    # Preparation for inference
    texts_process = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts_process,
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

if __name__ == "__main__":

    base_model_path = ""
    lora_path = None

    # 加载模型, 适用lora
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(base_model_path)
    if lora_path:
        model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        model = base_model
    model.eval()


    image_path = r""
    image = Image.open(image_path).convert("RGB")

    meme_text = "subway is similar to prostitution you pay someone else to do your wife's job"
    chat_text = chat_text_template.format(meme_text=meme_text)

    results = get_binary_probs_qwen_vl(model, processor, image_path, chat_text)

    for key, value in results.items():
        print(f"{key}: {value}")




























