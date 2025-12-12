import argparse
import gc
import sys
import warnings
sys.path.append('/remote-home/zhangkc/data/zhangkc/lmms-eval/lmms_eval/Qwen2_5_vl_refine/')
from modeling_selongvu_qwenvl25 import SEQwen2_5_VLForConditionalGeneration
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser, Qwen2VLForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from accelerate import Accelerator
import glob
import numpy as np
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import os

from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
from pathlib import Path
import random, copy
import json
from datasets import load_dataset

# 加载自己的模型


from torch.utils.data import Dataset

from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
import torchvision.transforms as T
from torchvision import transforms
from video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode


import importlib
from collections import defaultdict


from construct_prompt import construct_prompt
import re, subprocess
# # 设置可见gpu为3
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'


SEED = 24242424
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

prompt_templates = {
    "mistral": {
        "preprompt": "<s>[INST]",
        "postprompt": " [/INST]"
    },
    "vicuna": {
        "preprompt": "<s>A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER:",
        "postprompt": "ASSISTANT:"
    },
    "llama3": {
        "preprompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "postprompt": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "qwen2": {
        "preprompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
        "postprompt": "<|im_end|>\n<|im_start|>assistant\n",
    }, 
    "yi": {
        "preprompt": "<|im_start|>system\nAnswer the questions.<|im_end|>\n<|im_start|>user\n",
        "postprompt": "<|im_end|>\n<|im_start|>assistant\n",
    },
}  #postprompt是后缀，preprompt表示是总prompt的前缀  # 后面这个"<|im_end|>\n<|im_start|>assistant\n" 有什么用
# \nAnswer the question using a single word or phrase.
# The color of the bottle cap is
# answer = "Yellow"

# conv_qwen = Conversation(
#     system="""<|im_start|>system
# You are a helpful assistant.""",
#     roles=("<|im_start|>user", "<|im_start|>assistant"),
#     version="qwen",
#     messages=[],
#     offset=0,
#     sep_style=SeparatorStyle.CHATML,
#     sep="<|im_end|>",
# )


def safe_tokenize(tokenizer, text):
    tokenized = tokenizer.encode(text, return_tensors="pt")
    if tokenizer.bos_token != None and len(tokenized) > 0 and tokenized[0, 0] == tokenizer.bos_token_id:
        tokenized = tokenized[:, 1:]
    return tokenized

# 下面这个函数的作用是将answer_embeds拼接到input_embeds的后面，然后将input_embeds的长度补齐到accelerator.num_processes的倍数，然后将input_embeds分成accelerator.num_processes份，然后将每一份输入到模型中，然后将模型的输出
# 拼接起来，然后将拼接起来的输出再拼接起来，然后将拼接起来的输出和answer_ids进行比较，如果相等则返回1，否则返回0。
# answer = "more bet"
def eval_forward(accelerator, model, input_embeds, answer_embeds, pad_id, answer_ids, tokenizer):
    # first append answer_embeds to input_embeds
    prompt_length = input_embeds.shape[1]
    labels_length = answer_embeds.shape[1]
    input_embeds = torch.cat([input_embeds, answer_embeds], dim=1)
    # second pad input_embeds to the multiple of accelerator.num_processes
    pad_tensor = torch.tensor(
        [pad_id]
        * (
            (accelerator.num_processes * 2)
            - input_embeds.shape[1] % (accelerator.num_processes * 2)
        )
    ).unsqueeze(0).unsqueeze(-1).expand(-1, -1, input_embeds.shape[-1]).to(accelerator.device)  # 这个向量用于
    input_embeds = torch.cat([input_embeds, pad_tensor], dim=1)
    position_ids = (
        torch.arange(input_embeds.shape[1]).unsqueeze(0).expand(input_embeds.shape[0], -1)
    ).to(accelerator.device)
    print("accelerator的device是：", accelerator.device)
    accelerator.print(input_embeds.shape)
    prepared = prepare_seq_parallel_inputs(
        "zigzag_ring_attn",
        input_embeds,
        position_ids,
        None,
        accelerator.process_index,
        accelerator.num_processes,
        accelerator.device,
    )  # 
    local_input_embeds = prepared["local_input_ids"]
    local_position_ids = prepared["local_position_ids"]
    with torch.inference_mode():
        logits = model(
            inputs_embeds=local_input_embeds,
            position_ids=local_position_ids,
            use_cache=False,
        ).logits
        pred = logits.argmax(dim=-1)

    # gather all logits using accelerator.gather
    def undo_extract_local(gathered_value, world_size, dim=1):
        value_chunks = gathered_value.chunk(2 * world_size, dim=dim)
        reordered_chunks = [None] * (2 * world_size)
        for i in range(world_size):
            reordered_chunks[i] = value_chunks[i * 2]
            reordered_chunks[2 * world_size - i - 1] = value_chunks[i * 2 + 1]
        return torch.cat(reordered_chunks, dim=dim)

    correct = False

    gathered_logits = accelerator.gather(pred.squeeze(0)).unsqueeze(0)
    # undo extract local on the gathered logits
    pred = undo_extract_local(gathered_logits, accelerator.num_processes)
    pred = pred[:, prompt_length - 1 : prompt_length + labels_length - 1]
    # check if the logits are correct, extract argmax id
    # compare the predicted_ids with the labels
    correct = (pred == answer_ids.to(accelerator.device)).all()
    if  accelerator.is_main_process:
        print(
            "Predicted: ",
            tokenizer.decode(pred.squeeze().tolist()),
            "Answer: ",
            tokenizer.decode(answer_ids.squeeze().tolist()),
        )
        # print id as well
        print(
            "Predicted: ",
            pred.squeeze().tolist(),
            "Answer: ",
            answer_ids.squeeze().tolist(),
        )
    return int(correct)

# load_haystack这个函数的作用 是（将args.haystack_dir中的所有的.pt文件加载到内存中，然后将这些文件中的tensor拼接起来，然后返回拼接起来的tensor。）
def load_haystack(args, accelerator):
    haystack_embeddings = torch.load(f"{args.haystack_dir}/video_embeddings.pt").to(torch.bfloat16)
    # for file_path in tqdm(sorted(Path(args.haystack_dir).glob("*.pt"))[:args.max_frame_num], desc="Loading Haystack Embeddings...", disable=not accelerator.is_main_process):
    #     embeddings = torch.load(file_path, map_location="cpu").to(torch.bfloat16).unsqueeze(0)
    #     haystack_embeddings = embeddings if haystack_embeddings is None else torch.cat(
    #         [haystack_embeddings, embeddings], dim=0
    #     )
    return haystack_embeddings

# load_text_embeddings这个函数的作用是将str转换成token_ids，然后将token_ids转换成embeddings，然后返回embeddings。
# 这个函数 load_text_embeddings 用于加载文本的嵌入表示。它涉及几个步骤，包括令牌化文本、替换特定 token ID（可选）、以及获取文本嵌入。
"""该函数接受以下几个参数：

str: 要进行嵌入的文本字符串。
tokenizer: 用于将文本字符串转化为 token 的分词器。
model: 用于生成 token 嵌入的模型。
accelerator: 用于设备加速（如 GPU）管理的对象。
replace_double_newline: 一个布尔值，决定是否需要替换 token ID 271 为两个 198。
"""
def load_text_embeddings(str, tokenizer, model, accelerator, replace_double_newline=False): 
    token_ids = safe_tokenize(tokenizer, str) # 使用分词器 tokenizer 将输入字符串 str 转换为 token ID
    def replace_double_newline_func(token_ids):
        # replace_double_newline_func 函数查找所有 ID 为 271 的位置，并将这些位置的 token 替换为两个 198。替换双换行符函数
        # subsitute token id 271 to two 198]
        # for example:
        # from: tensor([[128000, 128006,   9125, 128007,    271,   2675,    527,    264,  11190, 4221,    323,  11376,  18328,     13]])
        # to: tensor([[128000, 128006,   9125, 128007,    198,    198,    2675,    527,    264,  11190, 4221,    323,  11376,  18328,     13]])
        # length will increase by number of 271
        double_newline_loc = (token_ids == 271).nonzero()[:, 1]  # 找到 token ID 为 271 的位置索引。
        double_newline_loc += torch.arange(len(double_newline_loc))  # 调整位置索引，因为每次替换后序列长度都会增大。
        if len(double_newline_loc) > 0:
            for loc in double_newline_loc:
                token_ids = torch.cat([token_ids[:, :loc], torch.tensor([[198, 198]]), token_ids[:, loc+1:]], dim=1)
        return token_ids
    if replace_double_newline:
        token_ids = replace_double_newline_func(token_ids)
    token_ids = token_ids.to(accelerator.device)  # 将 token IDs 转移到加速器管理的设备上，通常是 GPU。
    print("此处accelerator的device是：", accelerator.device, model.device)
    with torch.inference_mode():
        embeddings = model.model.embed_tokens(token_ids)
    return embeddings.to(torch.bfloat16) # 将嵌入转换为 bfloat16 数据类型，并返回。这种类型在保持较高精度的同时降低了内存需求。

def inference(args):
    model = args.model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        model_max_length=sys.maxsize,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    accelerator = Accelerator(
        mixed_precision="bf16",
        
    )
    kwargs = {"rope_theta": args.rope_theta} if args.rope_theta is not None else {}
    if "qwen2" in args.model.lower() or "longva" in args.model.lower():
        model = Qwen2ForCausalLM_RingAttn.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
            device_map=accelerator.device,
            **kwargs,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
            device_map=accelerator.device,
            **kwargs,
        )
    tokenizer.pad_token = tokenizer.eos_token
    # remember to remove <s>
    accelerator.print("Preparing Haystack...")
    #haystack_embeddings = load_haystack(args, accelerator)
    assert len(haystack_embeddings) >= args.max_frame_num, "Haystack embeddings are not enough. Max frame {} is not found. Currently only {} frames.".format(args.max_frame_num, len(haystack_embeddings))
    haystack_embeddings = haystack_embeddings[:args.max_frame_num].to(accelerator.device)
    prompt = prompt_templates[args.prompt_template]
    preprompt_embeddings = load_text_embeddings(prompt["preprompt"], tokenizer, model, accelerator, args.replace_double_newline)
    postprompt_embeddings = load_text_embeddings(prompt["postprompt"], tokenizer, model, accelerator, args.replace_double_newline)
    
    needle_dataset = load_dataset(args.needle_dataset)["test"]
    answer_embedding_list = []
    answer_id_list = []
    needle_embedding_list = []
    question_embeding_list = []
    for index, instance in enumerate(needle_dataset):
        answer = instance["answer"]
        question = instance["question"]
        needle_embedding_list.append(torch.load(args.needle_embedding_dir + f"/{index}.pt", map_location="cpu").to(torch.bfloat16).to(accelerator.device))
        answer_embedding_list.append(load_text_embeddings(answer, tokenizer, model, accelerator))
        answer_id_list.append(safe_tokenize(tokenizer, answer))
        question_embeding_list.append(load_text_embeddings(question, tokenizer, model, accelerator))
        
    accelerator.print("Starting Evaluation...")
    model = accelerator.prepare(model)
    model.gradient_checkpointing_enable()
    all_accuries = []
    for num_frames in tqdm(
        range(
            args.min_frame_num, args.max_frame_num + 1, args.frame_interval
        )
    ):
        for depth in np.arange(0, 1 + args.depth_interval, args.depth_interval):
            accuracies = []
            for question_embedding, needle_embedding, answer_embedding, answer_id in zip(question_embeding_list, needle_embedding_list, answer_embedding_list, answer_id_list):
                query_frame_idx = int(depth * num_frames)
                input_frames = torch.cat([haystack_embeddings[:query_frame_idx],needle_embedding.unsqueeze(0), haystack_embeddings[query_frame_idx:num_frames]], dim=0).view(-1, haystack_embeddings.shape[-1]).unsqueeze(0)
                input_emebds = torch.cat([preprompt_embeddings, input_frames,question_embedding, postprompt_embeddings], dim=1)
                correct = eval_forward(
                    accelerator, model, input_emebds, answer_embedding, tokenizer.pad_token_id, answer_id, tokenizer
                )
                gc.collect()
                torch.cuda.empty_cache()
                if accelerator.is_main_process:
                    accuracies.append(correct)
            if accelerator.is_main_process:
                result = {
                    "Num. Frame": num_frames,
                    "Frame Depth": round(depth * 100, -1),
                    "Score": sum(accuracies) / len(accuracies),
                }
                accelerator.print(result)
                all_accuries.append(result)
    if accelerator.is_main_process:
        model_name = args.model.split("/")[-1]
        os.makedirs(f"{args.output_path}/{model_name}", exist_ok=True)
        # save all_accuries as json
        with open(f"{args.output_path}/{model_name}/all_accuracies.json", "w") as f:
            json.dump(all_accuries, f, indent=4)
    return all_accuries, accelerator


def plot(args,  all_accuries):
    df = pd.DataFrame(all_accuries)
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["#F0496E", "#EBB839", "#9ad5b3"]
    )

    pivot_table = pd.pivot_table(
        df,
        values="Score",
        index=["Frame Depth", "Num. Frame"],
        aggfunc="mean",
    ).reset_index()  # This will aggregate
    pivot_table = pivot_table.pivot(
        index="Frame Depth", columns="Num. Frame", values="Score"
    )
    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    ax = sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        vmin=0,
        vmax=1,
        linecolor='white',
        linewidths=1.5, 
        cmap=cmap,
        cbar_kws={"label": "Score"},
    )
    
    # Set the color bar label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(14)
    cbar.ax.tick_params(labelsize=14)

    
    # Define the formatter function
    def thousands_formatter(x, pos):
        if x >= 1000:
            return f'{x/1000:.1f}K'
        return f'{x}'

    context_lengths = pivot_table.columns
    formatted_context_lengths = [thousands_formatter(x, None) for x in context_lengths]

    # More aesthetics
    plt.xlabel("Num. of Frames", fontsize=14)  # X-axis label
    plt.ylabel("Depth Percent", fontsize=14)  # Y-axis label
    plt.xticks(ticks=[i + 0.5 for i in range(len(context_lengths))], labels=formatted_context_lengths, rotation=45, fontsize=14)
    # plt.xticks(rotation=45, fontsize=14)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0, fontsize=14)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # save
    model_name = args.model.split("/")[-1]

    plt.savefig(f"{args.output_path}/{model_name}/heatmap.png")
    # calculate average accuracy
    average_accuracy = df["Score"].mean()
    print(f"Average Accuracy: {average_accuracy}")
    # save as txt
    with open(f"{args.output_path}/{model_name}/avg_accuracy.txt", "w") as f:
        f.write(f"Average Accuracy: {average_accuracy}\n")

# 以上函数都是原本longva为了测试 needle数据集的程序。 现在自己要测试videomme，所以修改和增删inference函数和加载数据集类如下。 这个类 copy from videochat2.
class MME_dataset(Dataset):
    def __init__(self, data_prefix, anno_path, num_segments=16, resolution=224, max_subtitle_len=4096):
        self.data_prefix = data_prefix
        with open(anno_path, 'r') as f:
            self.data_list = json.load(f)
            
        self.num_segments = num_segments
        self.max_subtitle_len = max_subtitle_len
        
        # transform   进行图像预处理，依次对图象进行缩放、裁剪、堆叠、标准化等操作
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
    
    def __str__(self):
        task_dict = {}
        total = 0
        for data in self.data_list:
            if data['duration_category'] not in ans_dict:
                task_dict[data['duration_category']] = {}
            for q in data['questions']:
                if q['task_type'] not in ans_dict[data['duration_category']]:
                    ans_dict[data['duration_category']][q['task_type']] = 0
                ans_dict[data['duration_category']][q['task_type']] += 1
                total += 1

        res = f"There are {len(self.data_list)} videos.\n"
        res += f"There are {total} QAs.\n"
        for k, v in task_dict.items():
            res += f"------{k}------\n"
            for kk, vv in task_dict.items():
                res += f"{kk}: {vv}\n"
                
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices

    def read_frame(self, video_path, bound=None):
        video_path = os.path.join(video_path, str(self.num_segments))
        
        if os.path.exists(video_path):
            frame_list = [p for p in os.listdir(video_path)]
        else:
            raise Exception
            
        images_group = list()
        
        for frame_name in frame_list:
            img = Image.open(os.path.join(video_path, frame_name))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())  #fromarray函数将numpy数组转换为PIL图像
            #     # 归一化到 [0, 1]
            # image_array = vr[frame_index].asnumpy()
            # image_normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min())

            # # 转换为 PIL 图像
            # image_pil = Image.fromarray((image_normalized * 255).astype(np.uint8))
            # images_group.append(image_pil)

            images_group.append(img)
        
        # # 为了试一下没有视频输入的情况  所以强行用随机值给下面的images——group 随机赋值
        # images_group = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(len(images_group))]
        
        return images_group
        # torch_imgs = self.transform(images_group)  # 好像longva要求的视频输出直接是 asnumpy 所以不进行这个转换了
        
        # return torch_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer = f"({answer}) {data['options'][ord(answer) - ord('A')][3:]}"
        for idx, c in enumerate(data['options']):
            cur_choice, cur_text = c[0], c[3:]
            question += f"({cur_choice}) {cur_text}\n"
        question = question.rstrip()
        return question, answer

    def __getitem__(self, idx):
        video_name = self.data_list[idx]['url'].split("watch?v=")[1]
        video_path = os.path.join(self.data_prefix, "data", video_name)   #  ！ 这个地方要根据自己的设置更改   # 因为作者把视频存储成16帧了 所以原来这个地方是frames，现在这个地方更改的是适合自己的
        video_path = video_path + '.mp4'
        # We store the videos with only 16 or 32 frames for testing,
        # since directly reading the whold videos cost a lot of time.
        # You can also read the whole video via self.read_video(video_path)
        torch_imgs = self.read_video(video_path)  # 读取视频，从中采帧，然后利用transform进行图像预处理（裁剪、归一化）
        duration_category = self.data_list[idx]['duration']
        qa_list = []
        #print(self.data_list[idx], idx)  #  {'video_id': '001', 'duration': 'short',  'url': 'https://www.youtube.com/watch?v=fFjv93ACGo8', 'videoID': 'fFjv93ACGo8'.}就是每一行..
        
        """ for qa in self.data_list[idx]:  # qa好像就是self.data_list[idx]  应该是个字典  是的，如上
            qa_list.append(self.qa_template(qa)) """

        """ same_video = []
        for video in self.data_list:
            video_id = video['video_id']
            if video_id == self.data_list[idx]['video_id'] and video not in same_video:
                
                same_video.append(video)

        for qa in same_video:  # qa好像就是self.data_list[idx]  应该是个字典  是的，如上
            qa_list.append(self.qa_template(qa)) """
        qa_list.append(self.qa_template(self.data_list[idx]))  
        subtitle = ""
        try:
            subtitle_path = os.path.join(self.data_prefix, "subtitle_vtt", video_name + ".vtt")   # 原来的程序为".vtt" 应该是支持网页视频字幕对应的 但是原数据集文件是 .srt文件 
            if os.path.exists(subtitle_path):
                subtitle = read_vtt_and_concatenate(subtitle_path, model.mistral_tokenizer, self.max_subtitle_len)
        except Exception:
            subtitle = ""
            print(f"Error for {subtitle_path}")
            
        return {
            'subtitle': subtitle,
            'video': torch_imgs, 
            'qa_list': qa_list,
            'duration_category': duration_category,
            'duration': duration_category,
            'video_id': video_name,
            'task_type': self.data_list[idx]['task_type'],
            'options': self.data_list[idx]['options'],
            'domain': self.data_list[idx]['domain'],
            'sub_category': self.data_list[idx]['sub_category'],


        }
    
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)

    if max_frames_num == 1:
        uniform_sampled_frames = np.array([np.random.choice(range(total_frame_num))])
    else:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)

    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)

def validate_choices(input_value, all_choices, input_name):
    if input_value == 'ALL':
        return all_choices
    else:
        selected_values = [item.strip() for item in input_value.split(",")]
        invalid_values = [item for item in selected_values if item not in all_choices]
        if invalid_values:
            raise ValueError(f"Invalid {input_name} type(s): {', '.join(invalid_values)}. "
                             f"Valid choices are: {', '.join(all_choices + ['ALL'])}")
        return selected_values

def inference_videomme(args):
    # 参照readme文件中快速调用的 代码改造 而来
    accelerator = Accelerator(
        mixed_precision="bf16",
        
    )
    DEFAULT_IMAGE_TOKEN = "<image>"
    device_map="cuda"
    num_frame = max_frames_num = args.max_frame_num  # you can change this to several thousands so long you GPU memory can handle it :)
    gen_kwargs = {"do_sample": False, "temperature": 1, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024} # 其中do_sample参数表示是否采样，这意味着模型将从概率分布中选择下一个 token；temperature参数表示采样温度，用于控制采样的多样性；top_p参数表示采样的 top-p 阈值，用于控制采样的多样性；num_beams参数表示束搜索的束大小，用于控制生成的多样性；use_cache参数表示是否使用缓存，用于加速生成；max_new_tokens参数表示生成的最大 token 数量，用于控制生成的长度。
    
    # you can also set the device map to auto to accomodate more frames
    
    bnb_model_from_pretrained_args = {} 
    if args.model_name == "qwenvl2":
        model = SEQwen2VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",   #  "sdpa"
            **bnb_model_from_pretrained_args
        )
    else:
        model = SEQwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",   #  "sdpa"
            **bnb_model_from_pretrained_args
        )
    
    processor = AutoProcessor.from_pretrained(args.model)
    # 设置 任务模型（暂定）
    # model.task = "video_train_caption"
    # model.task = "video_train_qa_with_text"
    model.task = "video_eval_qa_with_text"
    model= model.to(device_map)
    print(model)
   
    # # 输出模型结构和包含的参数
    # for name, param in model.named_parameters():
    #     print(name, param.size()) # 这个地方是正确的，所有参数大小都正确加载了
    #     print(param)  # 因为发生了梯度爆炸，所以我要看一下这些参数中哪些都有问题





    
    prompt = prompt_templates[args.prompt_template]
    videochat2_prompt = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
    videochat2_question_prompt="\nOnly give the best option."
    # # 上面那些是从videochat2中copy的， 下面是从作者自己构建的评估库 的imml的 task 的untils文件中复制过来的
    lmmseval_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    
    my_prompt = "The global and local features of each video frame will be provided to you later in chronological order. Carefully watch the video."
    if args.my_prompt:
        root_prompt = prompt["preprompt"] + my_prompt + lmmseval_prompt 
    else:
        root_prompt = prompt["preprompt"] + lmmseval_prompt  

    
    reasoning_types = [
        "count",
        "direction",
        "rotation",
        "shape&trend",
        "velocity&frequency",
        "visual_cues"
    ],
    demonstration_types = [
        "human",
        "object",
        "simulated"
    ]
    total_frames = max_frames_num
    # check reasoning types and demonstration types validity for tomato dataset
    reasoning_type = validate_choices(args.reasoning_type, reasoning_types, 'reasoning')
    demonstration_type = validate_choices(args.demonstration_type, demonstration_types, 'demonstration')
    # creat output directories & construct queries for each output path  从tomato数据集的github的评估函数中copy 过来的
    queries = defaultdict(list)
    existing_paths = list()
    
    if len(reasoning_type) == 1:
        reasoning_type = reasoning_type[0]
    print(f"Reasoning type: {reasoning_type}") 
    for rt in reasoning_type:
        dataset_path = f"/remote-home/zhangkc/A100_temp/TOMATO/data/{rt}.json"
        with open(dataset_path, "r") as f:
            qas = json.load(f)
        
        for dt in demonstration_type:
            # create output path
            output_subdir = '+'.join([rt, dt])
            output_path = args.output_dir + f"/tomato/{output_subdir}/{total_frames}.jsonl"
            curr_results = set()
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

            elif os.path.exists(output_path):
                existing_paths.append(output_path)
                with open(output_path, 'r') as f:
                    for line in f:
                        curr_results.add(json.loads(line)['id'])
            
            # construct query dictionary & leave out existing results
            for id_, qa in qas.items():
                if qa['demonstration_type'] == dt:
                    if curr_results and id_ in curr_results:
                        continue
                    qa['id'] = id_
                    print(qa)
                    queries[output_path].append(qa)
    
    if existing_paths:
        warnings.warn(f"Result json(s) {', '.join(existing_paths)} already exist! Will append new results to existing files", UserWarning)

    # generate responses 
    print("Generating responses ...")
    print("queries", queries)
    # data_dir = "/data/haohh/video-mme-bench/"  # 这个路径是视频的路径  
    # anno_path =  "/data/haohh/video-mme-bench/videomme/test-00000-of-00001.json" #"your_data_path/Video-MME.json" /data/haohh/haohh_file/mask_map_image.json
    # num_frame = max_frames_num
    # resolution = 224
    # dataset = MME_dataset(
    #     data_dir, 
    #     anno_path, 
    #     num_segments=num_frame, resolution=resolution
    # )

    # with open(anno_path, 'r') as f:
    #     res_json_data = json.load(f)
    
    output_name = args.output_name
    # answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    # # 如果路径不存在，创建文件
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)
    # if not os.path.exists(answers_file):
    #     # 如果文件不存在，则创建一个空文件
    #     with open(answers_file, 'w') as file:
    #         pass  # 这里不需要写入任何内容，只是为了创建文件
    # print(f"文件 {answers_file} 已创建。")
    # ans_file = open(answers_file, "w")
    
    correct = 0
    total = 0
    res_list = []
    acc_dict = {}
    answer_embedding_list = []
    answer_id_list = []
    video_embedding_list = []
    question_embeding_list = []
    accuracies = []
    
    
    i = 0
    for output_dir, qas in queries.items():
        
        
        print(f"Processing {output_dir} ... 正在处理任务")
        i += 1
        example_list = []
        answers_file = os.path.join(args.output_dir, f"{i}_{output_name}.json")
        # 如果路径不存在，创建文件
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            password = "!@#$%^+123456"
            # command = f"sudo -S chmod 777 {args.output_dir}"
            command = f"sudo -u dbcloud_admin -S chmod -R 777 {args.output_dir}"
            # subprocess.run(command, input=f"{password}\n", text=True, shell=True)
            with subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, text=True) as proc:
                proc.communicate(input="!@#$%^+123456\n")
        if not os.path.exists(answers_file):
            password = "!@#$%^+123456"
            # # command = f"sudo -S chmod 777 {answers_file}"
            # create_command = f"sudo -u dbcloud_admin touch {answers_file}"
            # subprocess.run(create_command, input=f"{password}\n", text=True, shell=True)
            # command = f"sudo -u dbcloud_admin -S chmod 777 {answers_file}"
            # subprocess.run(command, input=f"{password}\n", text=True, shell=True)

            # 创建文件的命令
            create_command = f"sudo -u dbcloud_admin touch {answers_file}"
            # 改变文件权限的命令
            chmod_command = f"sudo -u dbcloud_admin chmod -R 777 {args.output_dir}"

            # 执行创建文件的命令
            with subprocess.Popen(create_command, shell=True, stdin=subprocess.PIPE, text=True) as proc:
                proc.communicate(input="!@#$%^+123456\n")

            # 执行改变文件权限的命令
            with subprocess.Popen(chmod_command, shell=True, stdin=subprocess.PIPE, text=True) as proc:
                proc.communicate(input="!@#$%^+123456\n")
            
            # 如果文件不存在，则创建一个空文件
            with open(answers_file, 'w') as file:
                pass  # 这里不需要写入任何内容，只是为了创建文件
        print(f"文件 {answers_file} 已创建。")
        # ans_file = open(answers_file, "w")
        
        print(num_frame, "the current num frames setting")
        for query in tqdm(qas):
            # print(query)
            id_ = query['id']
            video_path = os.path.join('/remote-home/zhangkc/A100_temp/TOMATO/videos', query['demonstration_type'], '.'.join([query['key'], 'mp4']))
            question = query['question']
            options = query['options']
            optionized_list = [f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]
            gt = optionized_list[query['answer']]
            answer = gt
            print(video_path)

            prompt, all_choices, index2ans = construct_prompt(question=question,
                                                            options=options,
                                                            num_frames=total_frames)  
            final_prompt = prompt + "\nOnly give the best option."#+ "\n" + question  + "\n"
            
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": args.max_pixels, #360 * 420,
                            # "fps": 1.0,
                            "fps": 4.0,
                            # "max_frames": num_frame, # 这样会按照1/2/fps取帧这个和读取视频有关的 都在 smart_nframes 函数里面 可以通过指定这两个参数影响取帧.(qwen-vl-utils 库的vision_process.py 文件中)
                            "min_frames": 16,
                            "num_frames": num_frame, # 固定成均匀取帧
                        },
                        {
                            "type": "text",
                            "text": final_prompt
                        }
                    ]
                }
            ]

            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

            inputs = processor(
                    text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    # fps=fps, # 这个参数是专门针对视频设计的 
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs,
                )
            inputs = inputs.to("cuda")
                
            
            try:
                with torch.inference_mode():
                    if "with_text" in model.task:
                        extra_kwargs = {"question" : question , "idx":None, "k": args.k, "extra": args.extra, "input_data_text": question, "max_txt_len": 512} # 后面的参数是设置输入qformer的最大文本长度
                        
                        if "second_per_grid_ts" in inputs.keys():
                            inputs.pop("second_per_grid_ts")
                        
                        # 在qwenl25上居然会输出推理过程
                        outputs = model.generate(**inputs, max_new_tokens=16, **extra_kwargs) #, question=question ,k=8 
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
                        ]
                        # output_text = processor.batch_decode(
                        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        # )  #generate(input_ids, images=[video_tensor],  modalities=["video"], question=question ,k=8,  **gen_kwargs)
                    else:
                        outputs = model.generate(**inputs, max_new_tokens=256)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
                        ]
                        

                    
                output_text = processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        ) 
                #如果是list就取第一个
                if isinstance(output_text, list):
                    output_text = output_text[0]
                
                print(question ,"\n" , "output:", output_text, "\n", "answer:", answer)
                
                outputs = output_text
                
                # candidates = example["candidates"]
                # # 找到answer在candidates中的位置
                # answer_index = candidates.index(answer)
                
                # # 转换成大写字母
                # char_answer = chr(ord('A') + answer_index)
                # print(char_answer)
                
                # 按照统一标准进行对比处理
                # 首先对输出做正则匹配处理. 以下是longva的处理
                def extract_characters_regex(s):
                    s = s.strip()
                    answer_prefixes = [
                        "The best answer is",
                        "The correct answer is",
                        "The answer is",
                        "The answer",
                        "The best option is" "The correct option is",
                        "Best answer:" "Best option:",
                    ]
                    for answer_prefix in answer_prefixes:
                        s = s.replace(answer_prefix, "")

                    if len(s.split()) > 10 and not re.search("[ABCDEF]", s):
                        return ""

                    matches = re.search(r"[ABCDEF]", s)
                    if matches is None:
                        return ""
                    return matches[0]
                
                def extract_characters_regex_longvu(pred):
                    # 这是longvu的 处理
                    pred = pred.replace("Answer", "")

                    letters = ["A", "B", "C", "D", "E"]

                    pred_answer = re.findall("[\(\ \[]*([A-E])[\)\.\ \]]*", pred)

                    if pred_answer:
                        pred_answer = pred_answer[0].strip()
                        pred_answer = pred_answer.strip("()")
                    if pred_answer in letters:
                        pred_idx = letters.index(pred_answer)
                        pred = letters[pred_idx]
                    else:
                        print("pred_answer: ", pred_answer, " pred: ", pred, flush=True)
                        pred_idx = 2
                        pred = letters[pred_idx]
                    
                    return pred


                outputs_extra_longva = extract_characters_regex(outputs)
                outputs_extra_longvu = extract_characters_regex_longvu(outputs)
                
                
                if "answer to the question would be" in outputs:
                        
                        outputs = outputs.split("answer to the question would be")[-1]
                
                if "the correct answer is" in outputs:
                    
                    outputs = outputs.split("the correct answer is")[-1]
                
                # 去掉开头的空格和换行符
                outputs = outputs.replace("\n", "")
                outputs = outputs.strip()
                #
                # print(outputs[:4], answer[0] in outputs[:4])
                if outputs in answer or answer in outputs or outputs == answer  or outputs_extra_longva in answer or  outputs_extra_longvu in answer:  # 只取答案前半部分选项   or answer[:4] in outputs or answer[0] in outputs[:4]
                    accuracies.append(1)
                else:
                    accuracies.append(0)
                score = sum(accuracies) / len(accuracies)
                print(sum(accuracies), len(accuracies), "计算依据")
                print("运行到当前的score:", score)

                outputs = outputs.strip()
                
                new_example = {
                            "id": id_,
                            "question": question,
                            "response": outputs,
                            "all_choices": all_choices,
                            "index2ans": index2ans,
                            'gt': gt
                        }
                
                example_list.append(new_example)

            except:
                continue

        with open(output_dir, "w") as f:
            f.write(json.dumps(
                example_list, ensure_ascii=False
            ) + "\n")

            
        # ans_file.write(json.dumps(example_list, ensure_ascii=False) + "\n")
        # ans_file.flush()
        # ans_file.close()

        
    
def main(args):
    if args.plot_only:
        # load all_accuracies from json
        model_name = args.model.split("/")[-1]
        with open(f"{args.output_path}/{model_name}/all_accuracies.json", "r") as f:
            all_accuracies = json.load(f)
        plot(args, all_accuracies)
    else:
        inference_videomme(args)
        


if __name__ == "__main__":
    args = argparse.ArgumentParser() #/data/temp_zc/LongVA/longva/checkpoints/projectors_mytrain_7B1/llavanext-_data_temp_zc_clip-vit-large-patch14-336-_data_temp_zc_TinylQwen2-1.5B-Instruct-mlp2x_gelu-pretrain_blip558k_plain_1.5B
    args.add_argument("--model", type=str, default="/remote-home/zhangkc/A100_temp/Qwen2-VL-Finetune/checkpoints/my_train_total_model_finetune2_lr5e-5_epoch3_llm_conv_cadamw_gumbel_memory_251201_copy/checkpoint-22800")  #/data/temp_zc/haohh_file/checkpoint-41800 试一下自己训练的qwen2-1.5b的模型 /data/temp_zc/LongVA-7B 原本是预训练模型/data/temp_zc/LongVA-7B /data/temp_zc/LongVA/longva/checkpoints/finetune /data/temp_zc/LongVA/longva/checkpoints/my_train2/finetune_mymethod/checkpoint-200
    args.add_argument("--k", type=int, default=4)
    args.add_argument("--extra", type=int, default=8)
    args.add_argument("--model_name", type=str, default="selongvlmqwenvl25")
    args.add_argument("--max_pixels", type=int, default=360 * 420)
    args.add_argument("--my_prompt", default=True, action="store_true")
    args.add_argument('--reasoning_type', type=str, default='ALL')
    args.add_argument('--demonstration_type', type=str, default='ALL')
    args.add_argument("--max_frame_num", type=int, default=128) #  /data/temp_zc/LongVA/longva/checkpoints/my_train_total_model_finetune2_lr5e-5_epoch6/finetune_mymethod_llava_video_1.5B_qformer_independt_22
    args.add_argument("--needle_dataset", type=str, default="lmms-lab/v_niah_needles")
    args.add_argument("--min_frame_num", type=int, default=20)
    args.add_argument("--frame_interval", type=int, default=20)
    args.add_argument("--output_dir", type=str, default="mytest/selongvlmqwenvl25_tomato") # longva7b_output 是复现的训练好的原始模型论文结果 longva_qwen1.5b_eventbench  longva7b_eventbench
    args.add_argument("--output_name", type=str, default="qwenvl257b_tomato_256frames")
    args.add_argument("--depth_interval", type=float, default=0.1)  
    args.add_argument("--num_samples", type=int, default=1)
    args.add_argument("--rope_theta", type=float, default=None)
    
    args.add_argument("--prompt_template", default = "qwen2", type=str)
    args.add_argument("--replace_double_newline", action="store_true")
    args.add_argument("--plot_only", default= False ,action="store_true")
    
    main(args.parse_args())
