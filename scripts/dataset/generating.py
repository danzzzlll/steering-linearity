"""CLI script to generate slightly positive contrastive examples using vLLM.

Example usage:
    python generate_examples.py \
        --model_name Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8 \
        --num_iters 200 \
        --temperature 1.1 \
        --output_path /tmp/slightly_positive.pkl
"""

import argparse
import os
from tqdm import tqdm
import joblib
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from prompts import Prompts
from clean_text import contains_hieroglyphs


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate slightly positive contrastive examples with vLLM and save them with joblib." 
    )

    # Model / runtime settings
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
                        help="Model name or local path.")
    parser.add_argument("--quantization", type=str, default="gptq",
                        choices=["gptq", "awq", "none"],
                        help="Quantization method passed to vLLM.")
    parser.add_argument("--dtype", type=str, default="half",
                        help="Model dtype (half, bfloat16, float16, float32).")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="Number of tensor parallel GPUs.")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1,
                        help="Number of pipeline stages.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95,
                        help="Target fraction of GPU memory to use.")
    parser.add_argument("--cuda_visible_devices", type=str, default="0,1",
                        help="Comma‑separated list of CUDA device IDs to use.")

    # Generation hyper‑parameters
    parser.add_argument("--temperature", type=float, default=1.3,
                        help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top‑p nucleus sampling.")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Penalty for repeated tokens.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens to generate.")
    parser.add_argument("--n", type=int, default=5, help="Number of samples per call.")
    parser.add_argument("--best_of", type=int, default=7,
                        help="Number of candidates from which the top‑n are chosen.")

    # Loop / output
    parser.add_argument("--num_iters", "-N", type=int, default=100,
                        help="Total number of iterations (outer loop).")
    parser.add_argument("--save_every", type=int, default=50,
                        help="Dump to disk after this many iterations.")
    parser.add_argument("--output_path", type=str,
                        default="/kaggle/working/slightly_positive.pkl",
                        help="Path to save the generated list via joblib.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm = LLM(
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
        model=args.model_name,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        n=args.n,
        best_of=args.best_of,
    )

    prompt_text = "Generate one contrastive example with the exact structure below, no extra text."
    messages = [
        {"role": "system", "content": Prompts.system_prompt_slightly_positive},
        {"role": "user", "content": prompt_text},
    ]

    chat_template = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    results = []

    for idx in tqdm(range(args.num_iters), desc="Generating"):
        outputs = llm.generate([chat_template], sampling_params, use_tqdm=False)
        results.extend(
            [out.text for out in outputs[0].outputs if not contains_hieroglyphs(out.text)]
        )

        if (idx + 1) % args.save_every == 0:
            joblib.dump(results, args.output_path)
            print(f"[Iter {idx + 1}] Saved {len(results)} items → {args.output_path}")

    joblib.dump(results, args.output_path)
    print(f"Finished with {len(results)} items. Saved to {args.output_path}")


if __name__ == "__main__":
    main()
