import json
import os


NO_CHAT_DATASETS = {
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "repobench-p",
    "lcc",
}


def load_config(config_dir, config_name):
    config_path = os.path.join(config_dir, f"{config_name}.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_longbench_datasets(dataset_name, config_dir):
    dataset2prompt = load_config(config_dir, "dataset2prompt")
    dataset2maxlen = load_config(config_dir, "dataset2maxlen")
    dataset2samples = load_config(config_dir, "dataset2samples")
    supported = [
        dataset
        for dataset in dataset2samples
        if dataset in dataset2prompt and dataset in dataset2maxlen
    ]
    if dataset_name == "all":
        return supported
    if dataset_name not in supported:
        raise ValueError(
            f"Unsupported LongBench dataset: {dataset_name}. "
            f"Supported datasets: {', '.join(supported)}"
        )
    return [dataset_name]


def resolve_dataset_max_samples(dataset_name, config_dir, max_samples=None):
    configured = load_config(config_dir, "dataset2samples")[dataset_name]
    return configured if max_samples is None else min(configured, max_samples)


def should_use_chat_format(dataset_name):
    return dataset_name not in NO_CHAT_DATASETS


def build_prompt_records(rows, prompt_template, chat_formatter=None):
    records = []
    for row in rows:
        prompt = prompt_template.format(**row)
        if chat_formatter is not None:
            prompt = chat_formatter(prompt)
        records.append(
            {
                "prompt": prompt,
                "answers": row["answers"],
                "all_classes": row.get("all_classes", []),
            }
        )
    return records


def load_jsonl_rows(data_file):
    rows = []
    with open(data_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
