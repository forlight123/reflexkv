import argparse
import json
import os
import time
from enum import Enum
from pathlib import Path

from eval.backends import GenerationConfig, PromptRecord, get_backend_class
from eval.utils.eval_longbench import evaluate_file
from eval.utils.longbench import (
    build_prompt_records,
    load_config,
    load_jsonl_rows,
    resolve_dataset_max_samples,
    resolve_longbench_datasets,
    should_use_chat_format,
)


DEFAULT_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "config")


def _has_option(parser, option):
    return option in getattr(parser, "_option_string_actions", {})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run LongBench evaluation.")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--dataset", default="2wikimqa")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")

    known_args, _ = parser.parse_known_args(argv)
    backend_cls = get_backend_class(known_args.backend)
    parser = backend_cls.add_cli_args(parser)
    if not _has_option(parser, "--seed"):
        parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _ensure_file_exists(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {description}: {path}")


def _to_jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return _to_jsonable(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return str(value)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2, ensure_ascii=False)


def _append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _recover_jsonl_records(path):
    if not os.path.exists(path):
        return []

    valid_records = []
    valid_lines = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                valid_records.append(json.loads(line))
                valid_lines.append(line)
            except json.JSONDecodeError:
                break

    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(valid_lines)

    return valid_records


def _build_error_prediction(record, exc):
    return {
        "pred": "",
        "answers": record.answers,
        "all_classes": record.all_classes,
        "meta": {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        },
    }


def _normalize_prediction(record, payload):
    return {
        "pred": payload.get("pred", ""),
        "answers": payload.get("answers", record.answers),
        "all_classes": payload.get("all_classes", record.all_classes),
        "meta": payload.get("meta", {}),
    }


def _generate_batch_with_fallback(backend, records, gen_config):
    try:
        return [_normalize_prediction(record, payload) for record, payload in zip(records, backend.generate(records, gen_config))]
    except Exception as exc:
        if len(records) == 1:
            return [_build_error_prediction(records[0], exc)]

    outputs = []
    for record in records:
        try:
            payload = backend.generate([record], gen_config)[0]
            outputs.append(_normalize_prediction(record, payload))
        except Exception as single_exc:
            outputs.append(_build_error_prediction(record, single_exc))
    return outputs


def _prepare_prompt_records(dataset_name, rows, prompt_template, formatter):
    chat_formatter = formatter if should_use_chat_format(dataset_name) else None
    payloads = build_prompt_records(rows, prompt_template, chat_formatter=chat_formatter)
    return [
        PromptRecord(
            dataset=dataset_name,
            prompt=payload["prompt"],
            answers=payload["answers"],
            all_classes=payload["all_classes"],
        )
        for payload in payloads
    ]


def _resolve_save_dir(output_dir, dataset_name, run_name):
    if run_name:
        return os.path.join(output_dir, run_name, dataset_name)
    return os.path.join(output_dir, dataset_name)


def _run_single_dataset(args, backend, dataset_name, prompt_template, max_new_tokens, formatter):
    data_file = os.path.join(args.data_dir, f"{dataset_name}.jsonl")
    _ensure_file_exists(data_file, f"LongBench dataset file for {dataset_name}")

    rows = load_jsonl_rows(data_file)
    max_samples = resolve_dataset_max_samples(dataset_name, args.config_dir, args.max_samples)
    prompt_records = _prepare_prompt_records(
        dataset_name,
        rows[:max_samples],
        prompt_template,
        formatter,
    )

    save_dir = _resolve_save_dir(args.output_dir, dataset_name, args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    pred_file = os.path.join(save_dir, "pred.jsonl")
    run_config_file = os.path.join(save_dir, "run_config.json")
    run_summary_file = os.path.join(save_dir, "run_summary.json")

    completed = _recover_jsonl_records(pred_file) if args.resume else []
    if not args.resume and os.path.exists(pred_file):
        with open(pred_file, "w", encoding="utf-8"):
            pass

    _write_json(run_config_file, vars(args))

    remaining_records = prompt_records[len(completed) :]
    gen_config = GenerationConfig(max_new_tokens=max_new_tokens)
    start_time = time.time()

    for index in range(0, len(remaining_records), args.batch_size):
        batch = remaining_records[index : index + args.batch_size]
        predictions = _generate_batch_with_fallback(backend, batch, gen_config)
        for prediction in predictions:
            _append_jsonl(pred_file, prediction)

    score = float(evaluate_file(pred_file, dataset_name, args.config_dir))
    final_predictions = _recover_jsonl_records(pred_file)
    failed_predictions = sum(1 for item in final_predictions if item.get("pred", "") == "")
    _write_json(
        run_summary_file,
        {
            "dataset": dataset_name,
            "requested_samples": len(prompt_records),
            "completed_predictions": len(final_predictions),
            "failed_predictions": failed_predictions,
            "resumed_predictions": len(completed),
            "avg_score": score,
            "duration_seconds": round(time.time() - start_time, 4),
        },
    )


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    for config_name in (
        "dataset2prompt",
        "dataset2maxlen",
        "dataset2metric",
        "dataset2samples",
    ):
        _ensure_file_exists(
            os.path.join(args.config_dir, f"{config_name}.json"),
            f"LongBench config {config_name}.json",
        )

    datasets = resolve_longbench_datasets(args.dataset, args.config_dir)
    prompt_config = load_config(args.config_dir, "dataset2prompt")
    maxlen_config = load_config(args.config_dir, "dataset2maxlen")

    backend_cls = get_backend_class(args.backend)
    backend = backend_cls()
    try:
        backend.build(args)
        formatter = backend.get_prompt_formatter()
        for dataset_name in datasets:
            _run_single_dataset(
                args,
                backend,
                dataset_name,
                prompt_config[dataset_name],
                maxlen_config[dataset_name],
                formatter,
            )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
