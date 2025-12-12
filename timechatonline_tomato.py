import argparse
import json
import logging
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoProcessor

from qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Default parameters
RUN_NAME = "tomato_eval"
DROP_METHOD = "feature"
DROP_THRESHOLD = 0.5
DROP_ABSOLUTE = True
CKPT_PATH = "wyccccc/TimeChatOnline-7B"

TOMATO_DIR = "/remote-home/zhangkc/A100_temp/TOMATO/data"
VIDEO_DIR = "/remote-home/zhangkc/A100_temp/TOMATO/videos"
RESULT_DIR = "eval/result_tomato"
LOG_PATH = "log/{run_name}_{curr_time}.log"
OUTPUT_JSONL = "output/{run_name}_{curr_time}.jsonl"
DR_SAVE_PATH = "drop/{run_name}_{curr_time}.jsonl"

MIN_PIXELS = 360 * 420
MAX_PIXELS = 360 * 420
MAX_FRAMES = 32

# Prompt template
PROMPT = (
    "Select the best answer to the following multiple-choice question based on the video. "
    "Respond with only the letter (A, B, C, or D) of the correct option.\n{}\nOptions: {}\n"
    "The best answer is:"
)

# Dataset settings
REASONING_TYPES = [
    "count",
    "direction",
    "rotation",
    "shape&trend",
    "velocity&frequency",
    "visual_cues",
]
DEMONSTRATION_TYPES = ["human", "object", "simulated"]


def validate_choices(input_value, all_choices, input_name):
    if input_value == "ALL":
        return all_choices
    selected_values = [item.strip() for item in input_value.split(",")]
    invalid_values = [item for item in selected_values if item not in all_choices]
    if invalid_values:
        raise ValueError(
            f"Invalid {input_name} type(s): {', '.join(invalid_values)}. "
            f"Valid choices are: {', '.join(all_choices + ['ALL'])}"
        )
    return selected_values


def load_tomato_queries(tomato_dir, reasoning_type, demonstration_type):
    queries = []
    for rt in reasoning_type:
        dataset_path = osp.join(tomato_dir, f"{rt}.json")
        with open(dataset_path, "r") as f:
            qas = json.load(f)
        for dt in demonstration_type:
            for qid, qa in qas.items():
                if qa.get("demonstration_type") != dt:
                    continue
                qa["id"] = qid
                qa["reasoning_type"] = rt
                queries.append(qa)
    return queries


def normalize_video_path(video_dir, qa):
    video_path = qa.get("video_path") or qa.get("video") or qa.get("url")
    if video_path is None:
        raise ValueError(f"No video path found in qa entry: {qa}")
    if not osp.isabs(video_path):
        video_path = osp.join(video_dir, video_path)
    if not video_path.endswith(".mp4"):
        video_path = f"{video_path}.mp4"
    return video_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=RUN_NAME)
    parser.add_argument("--drop_method", type=str, default=DROP_METHOD)
    parser.add_argument("--drop_threshold", type=float, default=DROP_THRESHOLD)
    parser.add_argument("--drop_relative", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR)
    parser.add_argument("--tomato_dir", type=str, default=TOMATO_DIR)
    parser.add_argument("--video_dir", type=str, default=VIDEO_DIR)
    parser.add_argument("--reasoning_type", type=str, default="ALL")
    parser.add_argument("--demonstration_type", type=str, default="ALL")
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    args = parser.parse_args()

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_NAME = args.run_name
    DROP_METHOD = args.drop_method
    DROP_THRESHOLD = args.drop_threshold
    DROP_ABSOLUTE = not args.drop_relative
    CKPT_PATH = args.ckpt_path
    RESULT_DIR = args.result_dir
    TOMATO_DIR = args.tomato_dir
    VIDEO_DIR = args.video_dir
    MIN_PIXELS = args.min_pixels
    MAX_PIXELS = args.max_pixels
    MAX_FRAMES = args.max_frames

    LOG_PATH = osp.join(RESULT_DIR, LOG_PATH.format(run_name=RUN_NAME, curr_time=curr_time))
    OUTPUT_JSONL = osp.join(RESULT_DIR, OUTPUT_JSONL.format(run_name=RUN_NAME, curr_time=curr_time))
    DR_SAVE_PATH = osp.join(RESULT_DIR, DR_SAVE_PATH.format(run_name=RUN_NAME, curr_time=curr_time))

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, "output"), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, "drop"), exist_ok=True)
    os.makedirs(osp.join(RESULT_DIR, "log"), exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "%(asctime)s %(levelname)5s | %(message)s"
    fmt = logging.Formatter(fmt_str)
    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info(f"Running {RUN_NAME} on TOMATO")
    logger.info(f"Drop method: {DROP_METHOD}")
    logger.info(f"Drop threshold: {DROP_THRESHOLD}")
    logger.info("Drop absolute" if DROP_ABSOLUTE else "Drop relative")
    logger.info(f"Checkpoint path: {CKPT_PATH}")
    logger.info(f"Result dir: {RESULT_DIR}")
    logger.info(f"TOMATO dir: {TOMATO_DIR}")
    logger.info(f"Video dir: {VIDEO_DIR}")
    logger.info(f"Output jsonl: {OUTPUT_JSONL}")
    logger.info(f"Drop ratio info save path: {DR_SAVE_PATH}")
    logger.info(f"Min pixels: {MIN_PIXELS}")
    logger.info(f"Max pixels: {MAX_PIXELS}")
    logger.info(f"Max frames: {MAX_FRAMES}")

    reasoning_type = validate_choices(args.reasoning_type, REASONING_TYPES, "reasoning")
    demonstration_type = validate_choices(args.demonstration_type, DEMONSTRATION_TYPES, "demonstration")

    queries = load_tomato_queries(TOMATO_DIR, reasoning_type, demonstration_type)
    logger.info(f"Loaded {len(queries)} questions from TOMATO")

    torch.manual_seed(1234)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CKPT_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        CKPT_PATH,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    logger.info(f"Loaded model and processor from {CKPT_PATH}")

    cnt_total = defaultdict(int)
    cnt_correct = defaultdict(int)

    for qa in tqdm(queries, total=len(queries)):
        try:
            video_path = normalize_video_path(VIDEO_DIR, qa)
            options = qa.get("options", [])
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            reasoning = qa.get("reasoning_type", "unknown")
            demo_type = qa.get("demonstration_type", "unknown")

            message = {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "min_pixels": MIN_PIXELS,
                        "max_pixels": MAX_PIXELS,
                        "num_frames": MAX_FRAMES,
                    },
                    {
                        "type": "text",
                        "text": PROMPT.format(question, "\n".join(options)),
                    },
                ],
            }

            text = processor.apply_chat_template(
                [message], tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info([message])
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
                drop_method=DROP_METHOD,
                drop_threshold=DROP_THRESHOLD,
                drop_absolute=DROP_ABSOLUTE,
                dr_save_path=DR_SAVE_PATH,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = output_text[0]

            cnt_total["overall"] += 1
            cnt_total[reasoning] += 1
            key = f"{reasoning}_{demo_type}"
            cnt_total[key] += 1
            if isinstance(answer, str) and answer.strip().upper() in response:
                cnt_correct["overall"] += 1
                cnt_correct[reasoning] += 1
                cnt_correct[key] += 1

            output_dict = {
                "id": qa.get("id"),
                "reasoning_type": reasoning,
                "demonstration_type": demo_type,
                "question": question,
                "options": options,
                "answer": answer,
                "response": response,
                "video_path": video_path,
            }
            with open(OUTPUT_JSONL, "a" if osp.exists(OUTPUT_JSONL) else "w") as f:
                f.write(json.dumps(output_dict) + "\n")
        except Exception as e:
            logger.error(f"Error processing qa {qa.get('id', 'unknown')}: {e}")

    if cnt_total["overall"] == 0:
        logger.info("No question processed")
    else:
        overall_acc = 100 * cnt_correct["overall"] / cnt_total["overall"]
        logger.info(
            f"Total: {cnt_total['overall']}, Correct: {cnt_correct['overall']}, Accuracy: {overall_acc:.1f}%"
        )
    for rt in reasoning_type:
        if cnt_total[rt] == 0:
            continue
        acc = 100 * cnt_correct[rt] / cnt_total[rt]
        logger.info(f"Reasoning {rt}: {cnt_correct[rt]}/{cnt_total[rt]} = {acc:.1f}%")
    for rt in reasoning_type:
        for dt in demonstration_type:
            key = f"{rt}_{dt}"
            if cnt_total[key] == 0:
                continue
            acc = 100 * cnt_correct[key] / cnt_total[key]
            logger.info(f"{key}: {cnt_correct[key]}/{cnt_total[key]} = {acc:.1f}%")

    if DROP_METHOD is not None and DR_SAVE_PATH is not None and osp.exists(DR_SAVE_PATH):
        drop_list, total_list, ratio_list = [], [], []
        with open(DR_SAVE_PATH, "r") as f:
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

