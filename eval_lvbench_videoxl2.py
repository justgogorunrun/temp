import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from decord import VideoReader, cpu
from tqdm import tqdm

from videoxl2.videoxl2.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from videoxl2.videoxl2.conversation import SeparatorStyle, conv_templates
from videoxl2.videoxl2.model.builder import load_pretrained_model
from videoxl2.videoxl2.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)


def load_video(video_path: str, max_frames_num: int, fps: int = 1, max_fps: int | None = 4) -> Tuple[np.ndarray, List[float]]:
    """Load frames uniformly from a video while keeping an FPS upper bound."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    avg_fps = vr.get_avg_fps()
    effective_fps = avg_fps if avg_fps > 0 else fps

    step = round(effective_fps / fps) if fps > 0 else 1
    frame_idx = list(range(0, total_frame_num, step))

    if max_fps is not None:
        higher_fps = min(max_frames_num // max(len(frame_idx), 1), max_fps)
        if higher_fps > fps:
            higher_steps = round(effective_fps / higher_fps)
            frame_idx = list(range(0, total_frame_num, higher_steps))

    if len(frame_idx) > max_frames_num:
        frame_idx = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()

    if len(frame_idx) % 4 != 0:
        frame_idx = frame_idx[: -(len(frame_idx) % 4)]

    timestamps = [round(idx / effective_fps, 1) for idx in frame_idx]
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames, timestamps


def build_prompt(conv_template: str, question: str, num_frames: int, token_strategy: str) -> str:
    if conv_template == "llama_3":
        conv = conv_templates[conv_template].copy()
    else:
        conv = conv_templates[conv_template].copy()

    image_tokens = [DEFAULT_IMAGE_TOKEN] * (num_frames if token_strategy == "multiple" else 1)
    content = " ".join(image_tokens) + "\n" + question

    conv.append_message(conv.roles[0], content)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def prepare_time_embedding(times: List[float], tokenizer) -> List[torch.Tensor]:
    compress_frame = times[::4]
    time_embedding: List[torch.Tensor] = []
    for time in compress_frame:
        prefix = f"Time {time}s:"
        time_embedding.append(torch.tensor(tokenizer(prefix).input_ids, dtype=torch.long))
        time_embedding.append(torch.tensor([151654] * 144, dtype=torch.long))

    flat = torch.cat(time_embedding)
    return [flat]


def load_lvbench(data_root: str) -> List[dict]:
    video_dir = os.path.join(data_root, "videos", "00000")
    mapping_path = os.path.join(video_dir + ".json")
    meta_path = os.path.join(data_root, "video_info.meta.json")

    with open(mapping_path, "r") as f:
        video_name_find = json.load(f)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    records = []
    for data in meta:
        video_name_orin = data["key"]
        video_name_new = None
        for item in video_name_find:
            if video_name_orin == item["url"].split("watch?v=")[-1]:
                video_name_new = item["key"]
                break
        if video_name_new is None:
            continue

        video = f"{video_name_new}_h264.mp4"
        for qa in data["qa"]:
            question_orin = qa["question"]
            question = question_orin.split("\n")[0]
            question_format = f"Question: {question.capitalize()}\nOptions:\n"
            if "?\n" in qa["question"]:
                question_format += qa["question"].split("?\n")[1]
            else:
                question_format += qa["question"].split("\n", 1)[1]
            qa["question"] = question_format

        records.append({
            "key": data["key"],
            "type": data["type"],
            "video": video,
            "QA": data["qa"],
            "video_path": os.path.join(video_dir, video),
        })
    return records


def inference_lvbench(args):
    accelerator = Accelerator()
    device = accelerator.device

    model_name = get_model_name_from_path(args.model)
    llava_model_args = {
        "multimodal": True,
        "attn_implementation": args.attn_impl,
        "overwrite_config": {
            "mm_spatial_pool_stride": args.mm_spatial_pool_stride,
            "mm_spatial_pool_mode": args.mm_spatial_pool_mode,
        },
    }

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model,
        None,
        model_name,
        device_map=args.device_map,
        **llava_model_args,
    )
    model.eval()
    model.to(device)

    lvbench = load_lvbench(args.data_root)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.output_name}.jsonl")

    gen_kwargs = {
        "do_sample": True,
        "temperature": 0.5,
        "top_p": None,
        "num_beams": 1,
        "use_cache": True,
        "max_new_tokens": args.max_new_tokens,
    }

    with open(output_path, "w", encoding="utf-8") as writer:
        for example in tqdm(lvbench):
            if not os.path.exists(example["video_path"]):
                continue

            frames, timestamps = load_video(
                example["video_path"],
                args.max_frame_num,
                fps=args.fps,
                max_fps=args.max_fps,
            )
            frame_tensor = image_processor.preprocess(frames, return_tensors="pt")[
                "pixel_values"
            ].to(device, dtype=torch.float16)
            time_embedding = prepare_time_embedding(timestamps, tokenizer)

            stop_str = conv_templates[args.conv_template].sep
            if conv_templates[args.conv_template].sep_style == SeparatorStyle.TWO:
                stop_str = conv_templates[args.conv_template].sep2
            for qa in example["QA"]:
                final_prompt = args.root_prompt + "\n" + qa["question"] + "\n"
                prompt = build_prompt(
                    args.conv_template, final_prompt, frame_tensor.shape[1], args.token_strategy
                )
                input_ids = tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).to(device)

                pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                attention_masks = input_ids.ne(pad_token_id).to(device)
                stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

                model.config.mm_spatial_pool_stride = args.mm_spatial_pool_stride
                model.config.mm_spatial_pool_mode = args.mm_spatial_pool_mode

                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    images=[frame_tensor],
                    time_embedding=[t.to(device) for t in time_embedding],
                    modalities=["video"],
                    stopping_criteria=[stopping_criteria],
                    **gen_kwargs,
                )
                trimmed = outputs[:, input_ids.shape[1] :]
                text = tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
                qa["pred"] = text
                writer.write(json.dumps({"video": example["video"], "qa": qa}, ensure_ascii=False) + "\n")

    accelerator.print(f"Predictions saved to {output_path}")


def build_root_prompt(prompt_choice: str) -> str:
    videochat2_prompt = (
        "Carefully watch the video and pay attention to the cause and sequence of events, "
        "the detail and movement of objects, and the action and pose of persons. Based on your "
        "observations, select the best option that accurately addresses the question.\n"
    )
    videochat2_question_prompt = "\nOnly give the best option."
    lmmseval_prompt = (
        "Select the best answer to the following multiple-choice question based on the video and the subtitles. "
        "Respond with only the letter (A, B, C, or D) of the correct option."
    )
    if prompt_choice == "videochat2":
        return videochat2_prompt + videochat2_question_prompt
    return lmmseval_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="videoxl2 checkpoint path")
    parser.add_argument("--data_root", type=str, required=True, help="LVBench data root")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--output_name", type=str, default="videoxl2_lvbench")
    parser.add_argument("--conv_template", type=str, default="vicuna_v1")
    parser.add_argument("--token_strategy", type=str, default="single", choices=["single", "multiple"])
    parser.add_argument("--max_frame_num", type=int, default=128)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max_fps", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=2)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2")
    parser.add_argument("--device_map", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--prompt_style", type=str, default="videochat2", choices=["videochat2", "lmmseval"])
    parser.add_argument(
        "--root_prompt",
        type=str,
        default=None,
        help="If provided, overrides the prompt template selection.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.root_prompt = args.root_prompt or build_root_prompt(args.prompt_style)
    inference_lvbench(args)
