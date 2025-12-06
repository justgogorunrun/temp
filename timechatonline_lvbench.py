"""TimeChatOnline LVBench evaluation script.

This script mirrors the TimeChatOnline Video-MME pipeline but adapts it to
LVBench. Dataset loading logic follows the LVBench example so that video paths
and question formatting match the benchmark requirements.
"""

import argparse
import json
import logging
import os
import os.path as osp
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "..")))
from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration  # noqa: E402

# Default parameters copied from the Video-MME pipeline
RUN_NAME = "lvbench_feature_0d5"
DROP_METHOD = "feature"
DROP_THRESHOLD = 0.5
DROP_ABSOLUTE = True
CKPT_PATH = "wyccccc/TimeChatOnline-7B"

RESULT_DIR = "eval/result_lvbench"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
DR_SAVE_PATH = "drop/{run_name}_{curr_time}.jsonl"

MIN_PIXELS = 360 * 420
MAX_PIXELS = 360 * 420
NUM_FRAMES = 32

# LVBench paths (update if needed)
VIDEO_DIR = "/remote-home/zhangkc/data/temp_zc/LVBench/data/videos/00000"
VIDEO_MAP_JSON = "/remote-home/zhangkc/data/temp_zc/LVBench/data/videos/00000.json"
VIDEO_META_JSON = "/remote-home/zhangkc/data/temp_zc/LVBench/data/video_info.meta.json"

PROMPT = (
    "Select the best answer to the following multiple-choice question based on the "
    "video and the subtitles. Respond with only the letter (A, B, C, or D) of the "
    "correct option.\n{}"
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
FMT = logging.Formatter("%(asctime)s %(levelname)5s | %(message)s")


# Helper functions

def extract_characters_regex(s: str) -> str:
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def build_lvbench_entries(video_map_path: str, meta_path: str):
    with open(video_map_path, "r") as f:
        video_map = json.load(f)
    with open(meta_path, "r") as f:
        full_data = json.load(f)

    url_to_local = {item["url"].split("watch?v=")[-1]: item["key"] for item in video_map}

    entries = []
    for data in full_data:
        url_key = data["key"]
        if url_key not in url_to_local:
            continue
        video_local = url_to_local[url_key] + "_h264.mp4"
        for qa_item in data.get("qa", []):
            question_raw = qa_item["question"]
            question = question_raw.split("\n")[0]
            question_format = f"Question: {question.capitalize()}\nOptions:\n"
            if "?\n" in question_raw:
                question_format += question_raw.split("?\n")[1]
            else:
                question_format += question_raw.split("\n", 1)[1]
            entries.append(
                {
                    "video": video_local,
                    "question": question_format,
                    "answer": qa_item["answer"],
                    "type": data.get("type", ""),
                    "key": url_key,
                }
            )
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--drop_method", type=str, default=DROP_METHOD)
    parser.add_argument("--drop_threshold", type=float, default=DROP_THRESHOLD)
    parser.add_argument("--drop_relative", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--video_map_json", type=str, default=VIDEO_MAP_JSON)
    parser.add_argument("--video_meta_json", type=str, default=VIDEO_META_JSON)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES)
    args = parser.parse_args()

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name
    drop_method = args.drop_method
    drop_threshold = args.drop_threshold
    drop_absolute = not args.drop_relative

    result_dir = args.result_dir
    log_path = osp.join(result_dir, LOG_PATH.format(run_name=run_name, curr_time=curr_time))
    output_jsonl = osp.join(result_dir, OUTPUT_JSONL.format(run_name=run_name, curr_time=curr_time))
    dr_save_path = osp.join(result_dir, DR_SAVE_PATH.format(run_name=run_name, curr_time=curr_time))

    os.makedirs(osp.dirname(log_path), exist_ok=True)
    os.makedirs(osp.dirname(output_jsonl), exist_ok=True)
    os.makedirs(osp.dirname(dr_save_path), exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(FMT)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info("Running TimeChatOnline on LVBench")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Drop method: {drop_method}")
    logger.info(f"Drop threshold: {drop_threshold}")
    logger.info("Drop absolute" if drop_absolute else "Drop relative")
    logger.info(f"Checkpoint path: {args.ckpt_path}")
    logger.info(f"Result dir: {result_dir}")
    logger.info(f"Video dir: {args.video_dir}")
    logger.info(f"Video map json: {args.video_map_json}")
    logger.info(f"Video meta json: {args.video_meta_json}")
    logger.info(f"Output jsonl: {output_jsonl}")
    logger.info(f"Drop ratio save path: {dr_save_path}")
    logger.info(f"Min pixels: {args.min_pixels}")
    logger.info(f"Max pixels: {args.max_pixels}")
    logger.info(f"Num frames: {args.num_frames}")

    torch.manual_seed(1234)
    logger.info("Set manual seed to 1234")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        args.ckpt_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    logger.info(f"Loaded model and processor from {args.ckpt_path}")

    entries = build_lvbench_entries(args.video_map_json, args.video_meta_json)
    logger.info(f"Loaded {len(entries)} LVBench QA pairs")

    start_time = time.time()
    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)

    for entry in tqdm(entries):
        video_path = osp.join(args.video_dir, entry["video"])
        if not osp.exists(video_path):
            logger.warning(f"Video not found: {video_path}")
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "min_pixels": args.min_pixels,
                        "max_pixels": args.max_pixels,
                        "num_frames": args.num_frames,
                    },
                    {
                        "type": "text",
                        "text": PROMPT.format(entry["question"]),
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(torch.device("cuda"))

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            drop_method=drop_method,
            drop_threshold=drop_threshold,
            drop_absolute=drop_absolute,
            dr_save_path=dr_save_path,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = output_text[0]

        cnt_total["overall"] += 1
        if extract_characters_regex(response) == entry["answer"]:
            cnt_correct["overall"] += 1

        output_dict = {
            "video": entry["video"],
            "question": entry["question"],
            "answer": entry["answer"],
            "response": response,
            "type": entry["type"],
            "key": entry["key"],
        }
        with open(output_jsonl, "a" if osp.exists(output_jsonl) else "w") as f:
            f.write(json.dumps(output_dict) + "\n")

    end_time = time.time()
    cost_time = int(end_time - start_time)

    if cnt_total["overall"] == 0:
        logger.info("No question processed")
    else:
        acc = 100 * cnt_correct["overall"] / cnt_total["overall"]
        logger.info(
            f"Total: {cnt_total['overall']}, Correct: {cnt_correct['overall']}, Accuracy: {acc:.1f}%"
        )

    if drop_method is not None and dr_save_path is not None and osp.exists(dr_save_path):
        drop_list, total_list, ratio_list = [], [], []
        with open(dr_save_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            drop_ratio_info = json.loads(line)
            drop_list.append(drop_ratio_info.get("drop", 0))
            total_list.append(drop_ratio_info.get("total", 0))
            ratio_list.append(drop_ratio_info.get("ratio", 0))
        if sum(total_list) > 0 and ratio_list:
            total_dr = sum(drop_list) / sum(total_list)
            avg_dr = sum(ratio_list) / len(ratio_list)
            logger.info(f"Total drop ratio (weighted drop ratio): {100 * total_dr:.1f}%")
            logger.info(f"Average drop ratio (unweighted drop ratio): {100 * avg_dr:.1f}%")

    logger.info(
        f"Inference cost time: {cost_time // 3600}h {(cost_time % 3600) // 60}m {cost_time % 60}s"
    )


if __name__ == "__main__":
    main()
