"""Fine-tune the Hugging Face detector on a custom dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, cast
import numpy as np
import os
import torch
from datasets import Audio, DatasetDict, load_dataset  # type: ignore[import-not-found]
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the Deepfake detector on local audio clips.")
    parser.add_argument("--train-manifest", required=True, help="CSV file listing audio paths and labels for training.")
    parser.add_argument(
        "--eval-manifest",
        help="Optional CSV for validation. If omitted, 10% of the training manifest becomes the eval split.",
    )
    parser.add_argument("--audio-column", default="audio_path", help="Column containing absolute/relative audio paths.")
    parser.add_argument("--label-column", default="label", help="Column containing class labels (e.g., HUMAN/AI).")
    parser.add_argument(
        "--model-name-or-path",
        default="MelodyMachine/Deepfake-audio-detection-V2",
        help="Base checkpoint to start from (HF hub ID or local path).",
    )
    parser.add_argument("--output-dir", default="artifacts/finetuned-model", help="Where to store the fine-tuned weights.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Resample audio to this rate during preprocessing.")
    parser.add_argument("--num-epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--preprocessing-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _rename_column(dataset_dict: DatasetDict, old: str, new: str) -> DatasetDict:
    if old == new:
        return dataset_dict
    for split in dataset_dict.keys():
        if old in dataset_dict[split].column_names:
            dataset_dict[split] = dataset_dict[split].rename_column(old, new)
    return dataset_dict


def _resolve_audio_path(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path_str}")
    return str(path)


def _patch_optimizer_training_api() -> None:
    if not hasattr(torch.optim.Optimizer, "train"):
        torch.optim.Optimizer.train = lambda self: self  # type: ignore[attr-defined]
    if not hasattr(torch.optim.Optimizer, "eval"):
        torch.optim.Optimizer.eval = lambda self: self  # type: ignore[attr-defined]


def prepare_dataset(args: argparse.Namespace) -> tuple[DatasetDict, List[str]]:
    data_files = {"train": args.train_manifest}
    if args.eval_manifest:
        data_files["validation"] = args.eval_manifest

    dataset_generic = load_dataset("csv", data_files=data_files)
    if not isinstance(dataset_generic, DatasetDict):
        raise TypeError(
            "Expected load_dataset to return a DatasetDict. Provide both train and validation manifests to build multiple splits."
        )
    dataset = dataset_generic
    dataset = _rename_column(dataset, args.audio_column, "audio")
    dataset = _rename_column(dataset, args.label_column, "label")

    def _prepare_audio(batch: Dict[str, Any]) -> Dict[str, Any]:
        batch["audio"] = [{"path": _resolve_audio_path(path)} for path in batch["audio"]]
        return batch

    dataset = dataset.map(_prepare_audio, batched=True, load_from_cache_file=False)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=args.sample_rate))

    if "validation" not in dataset:
        split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)
        dataset = DatasetDict(train=split_dataset["train"], validation=split_dataset["test"])

    dataset = dataset.class_encode_column("label")
    label_list = dataset["train"].features["label"].names
    return dataset, label_list


def preprocess_dataset(
    dataset: DatasetDict, feature_extractor: AutoFeatureExtractor, args: argparse.Namespace
) -> DatasetDict:
    columns_to_keep = {"label"}
    columns_to_remove = [col for col in dataset["train"].column_names if col not in columns_to_keep]

    def _preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio_arrays = [audio["array"] for audio in batch["audio"]]
        extractor_call = cast(Callable[..., Dict[str, Any]], feature_extractor)
        inputs = extractor_call(audio_arrays, sampling_rate=args.sample_rate)
        result = {key: value for key, value in inputs.items()}
        result["label"] = batch["label"]
        return result

    return dataset.map(
        _preprocess,
        remove_columns=columns_to_remove,
        batched=True,
        batch_size=args.preprocessing_batch_size,
    )


def main() -> None:
    args = parse_args()
    os.environ["ACCELERATE_DISABLE_MPS_FALLBACK"] = "1"
    torch.manual_seed(args.seed)
    _patch_optimizer_training_api()

    dataset, labels = prepare_dataset(args)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)
    dataset = preprocess_dataset(dataset, feature_extractor, args)
    dataset.set_format(type="torch")

    id2label = {i: name for i, name in enumerate(labels)}
    label2id = {name: i for i, name in id2label.items()}

    model = AutoModelForAudioClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label={str(idx): name for idx, name in id2label.items()},
    )

    data_collator = DataCollatorWithPadding(tokenizer=feature_extractor, padding=True)

    def compute_metrics(predictions):
        preds = np.argmax(predictions.predictions, axis=1)
        accuracy = float((preds == predictions.label_ids).mean())
        return {"accuracy": accuracy}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        eval_steps=args.eval_steps,
        logging_steps=25,
        report_to=["none"],
        fp16=False,
        optim="adamw_torch",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    feature_extractor.save_pretrained(args.output_dir)

    with open(Path(args.output_dir) / "label_mapping.json", "w", encoding="utf-8") as fp:
        json.dump({"id2label": id2label, "label2id": label2id}, fp, indent=2)


if __name__ == "__main__":
    main()
