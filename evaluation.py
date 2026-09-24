from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from typing import List
import os
import re
# 和评估指标相关的函数


def eval_classification(gt: List[int], pred: List[int], target_names: List[str]):
    """
    评估分类结果并输出 (主要是单标签分类. )
    Args:
        y_true: 真实值列表
        y_pred: 预测值列表
        target_names: 对应的每个标签的名称列表, 例如 0: not-hateful, 1: hateful -> [not-hateful, hatefull]
    Returns: 分类报告、 混淆矩阵
    """
    label_list = [i for i in range(len(target_names))]
    report = classification_report(gt, pred, target_names=target_names, digits=4)
    confusion = confusion_matrix(gt, pred, labels=label_list)

    print('分类报告:\n', report)
    print('混淆矩阵:\n', confusion, '\n')
    return report, confusion


def save_classification_report(y_true: List[int], y_pred: List[int], target_names: List[str], save_path: str,
                               model_name: str, comment: str = ""):
    """
    计算真实值和预测值的分类得分, 并保存至本地
    Args:
        y_true: 真实值列表
        y_pred: 预测值列表
        target_names: 对应的每个标签的名称列表, 例如 0: not-hateful, 1: hateful -> [not-hateful, hateful]
        save_path: 保存路径
    """
    acc_report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            output_dict=True,
            digits=4
        )
    ).T

    # 将acc一行填空
    acc_report_df.iloc[-3, :2] = np.nan
    acc_report_df.iloc[-3, 3] = acc_report_df.iloc[-2, 3]

    # 添加模型名称和注释
    acc_report_df["model_name"] = model_name
    acc_report_df["comment"] = comment

    # 索引变成普通列
    acc_report_df = acc_report_df.reset_index()
    acc_report_df = acc_report_df.rename(columns={"index": "label"})

    # 把 model_name 放到最前面
    cols = ["model_name"] + [col for col in acc_report_df.columns if col != "model_name"]
    acc_report_df = acc_report_df[cols]
    acc_report_df = acc_report_df.round(4)

    if os.path.exists(save_path):
        before_output = pd.read_excel(save_path)
        before_output = before_output._append(acc_report_df, ignore_index=True)
        before_output.to_excel(save_path, index=False)
    else:
        acc_report_df.to_excel(save_path, index=False)

    print("测评结果成功保存在：", save_path)


def evaluate_fine_grained_classification(y_true, y_pred, all_classes: List[str]):
    """Calculate fine-grained evaluation metrics.
    计算多分类标签的评估
    此时标签传入的应该是一个列表，装着每个标签的预测字符. 示例如下:
    all_classes = ["race", "gender", "religion"]
    y_true = [
        ["race", "gender"],
        ["religion"],
        ["gender"]
    ]
    y_pred = [
        ["race"],
        ["religion", "gender"],
        ["gender"]
    ]
    """
    # Convert to binary format for multi-label evaluation

    mlb = MultiLabelBinarizer(classes=all_classes)
    binary_true = mlb.fit_transform(y_true)
    binary_pred = mlb.fit_transform(y_pred)

    acc = accuracy_score(binary_true, binary_pred)
    micro_f1 = f1_score(binary_true, binary_pred, average='micro')
    macro_f1 = f1_score(binary_true, binary_pred, average='macro')
    weighted_f1 = f1_score(binary_true, binary_pred, average='weighted')

    metrics_dict = {"accuracy": acc,
               "micro_f1": micro_f1,
               "macro_f1": macro_f1,
               "weighted_f1": weighted_f1}


    # Also return the classification report
    label_list = [i for i in range(len(all_classes))]
    report = classification_report(binary_true, binary_pred, target_names=all_classes, digits=4, zero_division=0)
    report_dict = classification_report(binary_true, binary_pred, target_names=all_classes, output_dict=True, digits=4, zero_division=0)
    confusion = multilabel_confusion_matrix(binary_true, binary_pred, labels=label_list)

    print('分类报告:\n', report)
    print('混淆矩阵:\n', confusion, '\n')
    print("评估指标:")
    print("accuracy: ", round_result(acc))
    print("micro_f1: ", round_result(micro_f1))
    print("macro_f1: ", round_result(macro_f1))
    print("weighted_f1: ", round_result(weighted_f1), '\n')

    return metrics_dict, report_dict


def save_fine_grained_classification_report(report, metrics: dict, save_path: str,
                               model_name: str, comment: str = ""):
    acc_report_df = pd.DataFrame(report).T

    # 添加 acc 一行
    acc_report_df.loc["accuracy"] = [np.nan, np.nan, round_result(metrics["accuracy"]), np.nan]

    # 添加模型名称和注释
    acc_report_df["model_name"] = model_name
    acc_report_df["comment"] = comment

    # 索引变成普通列
    acc_report_df = acc_report_df.reset_index()
    acc_report_df = acc_report_df.rename(columns={"index": "label"})

    # 把 model_name 放到最前面
    cols = ["model_name"] + [col for col in acc_report_df.columns if col != "model_name"]
    acc_report_df = acc_report_df[cols]
    acc_report_df = acc_report_df.round(4)


    if os.path.exists(save_path):
        before_output = pd.read_excel(save_path)
        before_output = before_output._append(acc_report_df, ignore_index=True)
        before_output.to_excel(save_path, index=False)
    else:
        acc_report_df.to_excel(save_path, index=False)

    print("测评结果成功保存在：", save_path)


def extract_think_answer(text: str):
    """
    从文本中提取 <think></think> 和 <answer></answer> 的内容。

    参数:
        text (str): 输入文本

    返回:
        dict: {
            "think": 提取到的 think 内容，没有则为 None,
            "answer": 提取到的 answer 内容，没有则为 None
        }
    """

    # re.DOTALL 表示 . 可以匹配换行
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

    think_content = think_match.group(1).strip() if think_match else ""
    answer_content = answer_match.group(1).strip() if answer_match else ""

    return {
        "think": think_content,
        "answer": answer_content
    }


def extract_prediction_rationale(text: str):
    """
    从如下格式文本中提取 Prediction 和 Rationale:

    Prediction: Yes/No
    Rationale: ...

    返回:
        {
            "prediction": ...,
            "rationale": ...
        }
    """
    prediction_match = re.search(r"Prediction:\s*(Yes|No)", text, re.IGNORECASE)
    rationale_match = re.search(r"Rationale:\s*(.*)", text, re.IGNORECASE | re.DOTALL)

    prediction = prediction_match.group(1).strip() if prediction_match else ""
    rationale = rationale_match.group(1).strip() if rationale_match else ""

    if prediction is not None:
        prediction = prediction.capitalize()  # 统一成 Yes / No

    return {
        "prediction": prediction,
        "rationale": rationale
    }


def round_result(result: float):
    """将结果转换为百分比得分形式，同时保留两位小数"""
    return round(result * 100, 2)



if __name__ == "__main__":
    y_true = [
        ["race", "gender"],
        ["religion"],
        ["gender"]
    ]
    y_pred = [
        ["race"],
        ["religion", "gender"],
        ["gender"]
    ]

    all_classes = ["race", "gender", "religion"]
    metrics, report =evaluate_fine_grained_classification(y_true, y_pred, all_classes)
    save_fine_grained_classification_report(report, metrics, "fine_grained_classification_report.xlsx", "ABC_MODEL", "test")









