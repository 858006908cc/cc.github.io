"""Command-line options and input loading shared by the attention experiments."""

import json
import re
from pathlib import Path


def dataset_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Use letters, digits, hyphens, or underscores for the dataset name.")
    return value


def add_input_options(parser):
    parser.add_argument("--dataset_name", required=True, type=dataset_name, help="Name shown in reports.")
    parser.add_argument("--examples", required=True, type=Path, help="JSONL file of examples.")
    parser.add_argument("--predictions", required=True, type=Path, help="Aligned NPZ of predicted labels.")
    parser.add_argument("--allow_pickle", action="store_true", help="Load object arrays from a trusted legacy NPZ.")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Hugging Face model ID or local Llama checkpoint.")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device, such as cuda:0 or cpu.")


def load_inputs(args):
    import numpy as np

    texts, labels = [], []
    with args.examples.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                example = json.loads(line)
                text, label = example["input"], example["label"]
                if not isinstance(text, str) or not isinstance(label, (str, int)):
                    raise ValueError("Expected a string input and a string or integer label.")
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{args.examples}:{line_number}: {error}") from error
            texts.append(text)
            labels.append(label)
    with np.load(args.predictions, allow_pickle=args.allow_pickle) as archive:
        predictions = archive["assigned_labels"]
    if predictions.ndim != 1 or len(predictions) != len(texts) or not texts:
        raise ValueError("Provide a nonempty JSONL and one predicted label per row in the same order.")
    return texts, labels, predictions


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select an available device with --device.")
        with torch.cuda.device(device):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation="eager")
    model.to(device).eval()
    return tokenizer, model
