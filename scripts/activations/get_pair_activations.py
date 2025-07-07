"""CLI to compute activations or steering vectors from paired prompts.

Examples
--------
# Dump self‑attention activations for every layer
python get_pair_activations.py \
    --pair_path /data/pairs.pkl \
    --output_path /tmp/acts.pkl

# Train steering vectors on layers 0‑3 for token −2
python get_pair_activations.py \
    --pair_path /data/pairs.pkl \
    --output_path /tmp/steering.pkl \
    --steering \
    --layers 0,1,2,3 \
    --token_index -2
"""

import argparse
import joblib
from typing import List, Union

from utils.prompt_builder import PromptTemplate
from utils.act_steer import (
    get_activations_from_pairs,
    get_steering_vector,
)
from utils.load_model import load_model


def _parse_layers(value: str) -> Union[int, List[int]]:
    """Convert layer CLI arg into int list or -1 (all layers)."""
    value = value.strip().lower()
    if value in {"-1", "all", "*"}:
        return -1
    try:
        return [int(v) for v in value.split(",") if v]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--layers expects comma‑separated ints or '-1/all'") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute activations or steering vectors for prompt pairs using a HuggingFace model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    parser.add_argument("--pair_path", required=True, help="Pickle file containing a list of raw prompt strings")
    parser.add_argument("--output_path", default="/tmp/activations.pkl", help="Where to save the resulting pickle")

    # Model
    parser.add_argument(
        "--model_name",
        default="NousResearch/Llama-2-7b-chat-hf",
        help="HuggingFace model name or local path",
    )

    # Mode toggle
    parser.add_argument(
        "--steering",
        action="store_true",
        help="If set, trains steering vectors instead of just extracting activations",
    )

    # Generation / extraction hyper‑params
    parser.add_argument(
        "--layers",
        type=_parse_layers,
        default="-1",
        help="Comma‑separated layer indices, or -1/all for every layer",
    )
    parser.add_argument(
        "--layer_type",
        choices=["self_attn", "mlp"],
        default="self_attn",
        help="Which sub‑module to read activations from (ignored for steering mode)",
    )
    parser.add_argument(
        "--token_index",
        type=int,
        default=-2,
        help="Token position used when training steering vectors",
    )

    return parser.parse_args()


def make_pair(example: str):
    """Build paired prompts with a shared system prefix."""
    pt = PromptTemplate(
        system_prompt=(
            "You are an AI red‑teaming expert creating high‑stakes evaluation data. "
            "Your task is to generate examples to test a model's ability to remain calm "
            "and confident under pressure."
        )
    )
    pt.add_user_message(example)
    base = pt.build_prompt()
    return (base + " (A)", base + " (B)")


def main() -> None:
    args = parse_args()

    # Load raw examples
    raw_examples: List[str] = joblib.load(args.pair_path)
    data_pairs = [make_pair(x) for x in raw_examples]

    # Load model and tokenizer
    tokenizer, model = load_model(args.model_name)

    # Either extract activations or train steering vectors
    if args.steering:
        get_steering_vector(
            model=model,
            tokenizer=tokenizer,
            data_pairs=data_pairs,
            layers=args.layers,
            path_to_save=args.output_path,
            token_index=args.token_index,
        )
    else:
        get_activations_from_pairs(
            model=model,
            tokenizer=tokenizer,
            data_pairs=data_pairs,
            path_to_save=args.output_path,
            layers=args.layers,
            layer_type=args.layer_type,
        )


if __name__ == "__main__":
    main()
