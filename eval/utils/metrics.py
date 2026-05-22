import re
import string
from collections import Counter

import jieba
from fuzzywuzzy import fuzz
from rouge import Rouge


def normalize_answer(text):
    """Lower text and remove punctuation, articles, and extra whitespace."""

    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value):
        return " ".join(value.split())

    def remove_punc(value):
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def normalize_zh_answer(text):
    """Lower text and remove punctuation and extra whitespace."""

    def white_space_fix(value):
        return "".join(value.split())

    def remove_punc(value):
        cn_punctuation = (
            "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～"
            "｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        )
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in value if ch not in all_punctuation)

    return white_space_fix(remove_punc(text.lower()))


def count_score(prediction, ground_truth, **kwargs):
    del kwargs
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth))
    return float(0.0 if not numbers else right_num / len(numbers))


def retrieval_score(prediction, ground_truth, **kwargs):
    del kwargs
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth_id))
    return float(0.0 if not numbers else right_num / len(numbers))


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    del kwargs
    matches = re.findall(r"段落(\d+)", ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth_id))
    return float(0.0 if not numbers else right_num / len(numbers))


def code_sim_score(prediction, ground_truth, **kwargs):
    del kwargs
    all_lines = prediction.lstrip("\n").split("\n")
    candidate = ""
    for line in all_lines:
        if "`" not in line and "#" not in line and "//" not in line:
            candidate = line
            break
    return fuzz.ratio(candidate, ground_truth) / 100


def classification_score(prediction, ground_truth, **kwargs):
    all_classes = kwargs["all_classes"]
    matches = [class_name for class_name in all_classes if class_name in prediction]
    filtered = [
        match
        for match in matches
        if not (match in ground_truth and match != ground_truth)
    ]
    if ground_truth in filtered:
        return 1.0 / len(filtered)
    return 0.0


def rouge_score(prediction, ground_truth, **kwargs):
    del kwargs
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs):
    del kwargs
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    return rouge_score(prediction, ground_truth)


def f1_score(prediction, ground_truth, **kwargs):
    del kwargs
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction, ground_truth, **kwargs):
    return f1_score(
        normalize_answer(prediction).split(),
        normalize_answer(ground_truth).split(),
    )


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if token]
    ground_truth_tokens = [token for token in ground_truth_tokens if token]
    return f1_score(prediction_tokens, ground_truth_tokens)
