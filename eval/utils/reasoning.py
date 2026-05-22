import json
from pathlib import Path


_PROMPT_FIELDS = ("problem", "question", "input", "prompt", "text")
_ANSWER_FIELDS = ("answer", "answers")
_META_FIELDS = ("unique_id", "id", "subject", "level")


def load_config(config_dir, config_name):
    config_path = Path(config_dir) / f"{config_name}.json"
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_reasoning_datasets(dataset_name, config_dir):
    dataset2prompt = load_config(config_dir, "reasoning_dataset2prompt")
    dataset2maxlen = load_config(config_dir, "reasoning_dataset2maxlen")
    dataset2samples = load_config(config_dir, "reasoning_dataset2samples")
    supported = [
        dataset
        for dataset in dataset2samples
        if dataset in dataset2prompt and dataset in dataset2maxlen
    ]
    if dataset_name == "all":
        return supported
    if dataset_name not in supported:
        raise ValueError(
            f"Unsupported reasoning dataset: {dataset_name}. "
            f"Supported datasets: {', '.join(supported)}"
        )
    return [dataset_name]


def resolve_reasoning_max_samples(dataset_name, config_dir, max_samples=None):
    configured = load_config(config_dir, "reasoning_dataset2samples")[dataset_name]
    return configured if max_samples is None else min(configured, max_samples)


def load_reasoning_rows(data_file):
    rows = []
    with open(data_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _normalize_answers(row):
    for key in _ANSWER_FIELDS:
        if key in row:
            answers = row[key]
            if isinstance(answers, list):
                return [str(answer) for answer in answers]
            return [str(answers)]
    raise KeyError("Missing answer field in reasoning row")


def build_reasoning_prompt_context(row):
    context = dict(row)
    prompt_value = None
    for key in _PROMPT_FIELDS:
        if key in row and row[key] is not None:
            prompt_value = row[key]
            break
    if prompt_value is None:
        raise KeyError("Missing prompt-bearing field in reasoning row")
    for key in _PROMPT_FIELDS:
        context[key] = prompt_value
    return context


def _build_meta(row):
    meta = {}
    for key in _META_FIELDS:
        if key in row and row[key] is not None:
            meta[key] = row[key]
    return meta


def build_reasoning_prompt_records(dataset_name, rows, prompt_template, chat_formatter=None):
    del dataset_name
    records = []
    for row in rows:
        context = build_reasoning_prompt_context(row)
        prompt = prompt_template.replace("\\boxed{}", "\\boxed{{}}").format(**context)
        if chat_formatter is not None:
            prompt = chat_formatter(prompt)
        record = {
            "prompt": prompt,
            "answers": _normalize_answers(row),
            "all_classes": row.get("all_classes", []),
            "meta": _build_meta(row),
        }
        records.append(record)
    return records
