import argparse
import os
import re
from typing import List, Optional

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm

from qwen_vl_utils import process_vision_info

from dataset.load_MMR_V import load_MMR_V
from utils import read_json, write_to_json

PROMPT_TEMPLATE = """
[[INSTRUCTIONS]]
Please select the best answer to the following multiple-choice question based on the video.
Only one option is the most accurate answer in relation to the question and the video.

What is the correct answer to this question [[QUESTION]]
Options:
[[OPTIONS]]

Let's think step by step.
[[END OF INSTRUCTIONS]]
[[QUESTION]]
{question}
[[END OF QUESTION]]
[[OPTIONS]]
{options}
[[END OF OPTIONS]]
[[OUTPUT FORMAT]]
Format your answer as follows:
[Analyze the best option for question]
[Justification for your final choice based on the thinking process.]

Give the final correct option number in the following format: "[[A]]" or "[[B]]" or "[[C]]" or "[[D]]" ...
[[END OF OUTPUT FORMAT]]
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MMR-V evaluation with local QwenVL series models."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local path or HF repo for QwenVL model weights.",
    )
    parser.add_argument(
        "--model-type",
        choices=["qwen2vl", "qwen2.5vl"],
        default="qwen2.5vl",
        help="Select Qwen2VL or Qwen2.5VL model class.",
    )
    parser.add_argument(
        "--annotation-path",
        default=None,
        help="Optional JSON annotation path; if omitted, uses load_MMR_V().",
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Directory that stores MMR-V videos.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output JSON path for predictions.",
    )
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=448*448)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--attn-impl", default="flash_attention_2")
    return parser.parse_args()


def load_samples(annotation_path: Optional[str]) -> List[dict]:
    if annotation_path:
        return read_json(annotation_path)
    return load_MMR_V()


def build_model(model_path: str, model_type: str, attn_impl: str):
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": attn_impl,
    }
    if model_type == "qwen2vl":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs,
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs,
        )
    return model


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = load_samples(args.annotation_path)
    model = build_model(args.model_path, args.model_type, args.attn_impl).to(device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    results = []
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    for idx, sample in enumerate(tqdm(samples, desc="MMR-V")):
        video_path = os.path.join(args.video_dir, sample["video"])
        question = sample["question"]
        options = sample["options"]
        prompt = PROMPT_TEMPLATE.format(question=question, options=options)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_frames": args.max_frames,
                        "max_pixels": args.max_pixels,
                        "fps": args.fps,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        chat_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
        )
        inputs = processor(
            text=[chat_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        sample_output = dict(sample)
        sample_output["model_raw_response"] = output_text

        match = re.search(r"\[\[([A-Z])\]\]", output_text)
        if match:
            sample_output["model_response"] = {"final_answer": match.group(1)}
        else:
            sample_output["model_response"] = {"final_answer": None}

        results.append(sample_output)
        write_to_json(results, args.output_path, indent=4)

        if idx == 0:
            print(output_text)


if __name__ == "__main__":
    main()
