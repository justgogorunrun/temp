# 这个文件是用于测试qwenvl系列模型，包括qwen2vl、qwen2.5vl、qwen3vl系列模型的范例文件，以mlvu为例子给出这个系列的模型的推理pipeline。
import argparse
import gc
import sys,re

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser
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
import random
import json
from datasets import load_dataset
import subprocess

# apply_seq_parallel_monkey_patch("zigzag_ring_attn", "llama")
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

class MLVU_dataset(Dataset):
    def __init__(self, data_prefix, anno_path, sub_task, num_segments=16, resolution=224, max_subtitle_len=4096):
        self.data_prefix = data_prefix
        with open(anno_path, 'r') as f:
            self.data_list = json.load(f)
            
        self.num_segments = num_segments
        self.max_subtitle_len = max_subtitle_len
        self.sub_task = sub_task + "/"
        
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

    def caption_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Summary:\n"
        
        answer = "" # 这个mlvu的数据集暂时没有提供答案，所以这里先不管
        return question
    
    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        # answer = f"({answer}) {data['candidates'][ord(answer) - ord('A')][3:]}"
        # for idx, c in enumerate(data['candidates']):
        #     cur_choice, cur_text = c[0], c[3:]
        #     question += f"({cur_choice}) {cur_text}\n"
        # question = question.rstrip()

        # 首先要将选项和答案带上字母ABCDEF
        for idx, c in enumerate(data['candidates']):
            
            option_letter = chr(ord('A') + idx)  # 将索引转换为字母（A、B、C、D...）
            data['candidates'][idx] = f"({option_letter}) {c}"  # 将选项和答案带上字母ABCDEF
            option = data['candidates'][idx]

            if c == answer:
                answer = f"({option_letter}) {answer}"  # 将答案带上正确的字母ABCDEF

            question += f"{option}\n"

        return question, answer

    def __getitem__(self, idx):
        video_name = self.data_list[idx]["video"]
        # /remote-home/zhangkc/data/temp_zc/MLVU/MLVU/video/2_needle/001.mp4  如果只当子任务的json文件 这个地方要改成对应的
        # video_path = os.path.join(self.data_prefix, self.sub_task, "/")   #  ！ 这个地方要根据自己的设置更改   # 因为作者把视频存储成16帧了 所以原来这个地方是frames，现在这个地方更改的是适合自己的
        # video_path  = video_path + video_name
        # video_path = os.path.join(self.data_prefix, "4_count/", video_name)
        video_path = os.path.join(self.data_prefix, self.sub_task, video_name)
        # We store the videos with only 16 or 32 frames for testing,
        # since directly reading the whold videos cost a lot of time.
        # You can also read the whole video via self.read_video(video_path)
        # # 因为qwen2vl自带视频加载处理器 所以这个地方不需要额外加载视频
        # torch_imgs = self.read_video(video_path)  # 读取视频，从中采帧，然后利用transform进行图像预处理（裁剪、归一化）
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
        # qa_list.append(self.caption_template(self.data_list[idx]))  
        # print(self.qa_template(self.data_list[idx]))
        qa_list.append(self.qa_template(self.data_list[idx]))  # 当前回答的 json子任务是 mlvu的选择题，所以这里调用的是qa_template函数
        
            
        return {
            
            'video_path': video_path,
            # 'video': torch_imgs, 
            'qa_list': qa_list,
            'duration_category': duration_category,
            'video_id': video_name,
            "question_type": self.data_list[idx]["question_type"]
        }

    


def inference_mlvu(args):
    # 参照readme文件中快速调用的 代码改造 而来
    accelerator = Accelerator(
        mixed_precision="bf16",
        
    )
    DEFAULT_IMAGE_TOKEN = "<image>"
    device_map="cuda"
    max_frames_num = args.max_frame_num  # you can change this to several thousands so long you GPU memory can handle it :)
    gen_kwargs = {"do_sample": True, "temperature": 0.5, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}
    
    # you can also set the device map to auto to accomodate more frames
    bnb_model_from_pretrained_args = {} 
    
    if args.model_name == "qwenvl2":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",   #  "sdpa"
                **bnb_model_from_pretrained_args
            )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",   #  "sdpa"
                **bnb_model_from_pretrained_args
            )

    
    processor = AutoProcessor.from_pretrained(args.model)
    # 设置 任务模型（暂定）
    # model.task = "video_train_caption"
    # model.task = "video_train_qa_with_text"
    model= model.to(device_map)
    
    print(model)
   
    # # 输出模型结构和包含的参数
    # for name, param in model.named_parameters():
    #     print(name, param.size()) # 这个地方是正确的，所有参数大小都正确加载了


            
    
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
    root_caption_prompt = my_prompt + "Respond with only the letter (A, B, C, or D) of the correct option.\n" #+ " Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, give the detailed caption of this video.\n"

    
    # data_dir = "/data/temp_zc/MLVU/MLVU/video/"  # 这个路径是视频的路径  
    data_dir = "/remote-home/zhangkc/data/temp_zc/MLVU/MLVU/video/"  # 这个路径是视频的路径  

    # anno_path =  "/remote-home/zhangkc/data/temp_zc/MLVU/MLVU/json/4_count.json"  #"/remote-home/zhangkc/data/temp_zc/MLVU/MLVU/json/2_needle.json" #"your_data_path/Video-MME.json" (一个一个单独测试吧)
    anno_path = args.anno_path

    # 从anno_path 中截取子任务的 名字
    sub_task = anno_path.split("/")[-1].split(".")[0]
    num_frame = max_frames_num
    resolution = 336  # 这个参数 会对结果造成很大影响！（之前不小心全部压缩成224了，稍微调大一点）
    dataset = MLVU_dataset(
        data_dir, 
        anno_path, 
        sub_task = sub_task,
        num_segments=num_frame, resolution=resolution
    )

    with open(anno_path, 'r') as f:
        res_json_data = json.load(f)
    
    output_name = args.output_name
    # if sub_task 参数不是none  那么将文件名在 最后加上
    if args.sub_task:
        output_name = output_name + "_" + sub_task

    answers_file = os.path.join(args.output_dir, f"{output_name}.json")

    

       
    # 如果路径不存在，创建文件
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        # 同时在这个地方以dbcloud_admin 用户的身份修改 创建文件  的权限为 777. （要求输入密码的话自动输入 !@#$%^+123456）
        # os.system(f"sudo chmod 777 {args.output_dir}")
        password = "!@#$%^+123456"
        # command = f"sudo -S chmod 777 {args.output_dir}"
        command = f"sudo -u dbcloud_admin -S chmod -R 777 {args.output_dir}"
        # subprocess.run(command, input=f"{password}\n", text=True, shell=True)
        with subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, text=True) as proc:
            proc.communicate(input="!@#$%^+123456\n")
    if not os.path.exists(answers_file):
        password = "!@#$%^+123456"
        

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
    ans_file = open(answers_file, "w")
    
    correct = 0
    total = 0
    res_list = []
    acc_dict = {}
    answer_embedding_list = []
    answer_id_list = []
    video_embedding_list = []
    question_embeding_list = []
    accuracies = []
    for idx, example in enumerate(tqdm(dataset)):
        
        qa_count = len(example['qa_list'])
        #acc_dict[duration_category][1] += qa_count
        total += qa_count  
        video_path = example["video_path"]
        # TC, H, W = video.shape
        # video = video.reshape(1, TC//3, 3, H, W).to("cuda")   
        # video = video.squeeze(0)   
        # answer = example["qa_list"][0][1]
        print(example["qa_list"])
        question = example["qa_list"][0][0] # 当前数据条目的 问题 视频等信息 其中MM_dataset 类中已经处理了让问题中加上对应选项信息
        answer = example["qa_list"][0][1]
        
        
        final_prompt = root_caption_prompt  + "\n" + question  + "\n" 

        # print(final_prompt)
        # 因为训练完有一点不遵循指令了，所以改一下后面部分，强调一下只输出选项
        #final_prompt = root_prompt + DEFAULT_IMAGE_TOKEN + "\n" + question  + "\n" +  "<|im_end|>\n<|im_start|>assistant\n" + "The correct option is:\n"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                        "max_frames": num_frame, # 这个和读取视频有关的 都在 smart_nframes 函数里面 可以通过指定这两个参数影响取帧.(qwen-vl-utils 库的vision_process.py 文件中)
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
            
        
        with torch.inference_mode():
            if "with_text" in model.task:
                extra_kwargs = {"question" : question , "idx":None, "k": 8, "extra": 8 , "input_data_text": question, "max_txt_len": 512} # 后面的参数是设置输入qformer的最大文本长度
                outputs = model.generate(**inputs, max_new_tokens=256) # , **extra_kwargs
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
        # print(question ,"\n" , "output:", output_text, "\n", "answer:", answer)
        # outputs = output_text
        #如果是list就取第一个
        if isinstance(output_text, list):
            output_text = output_text[0]
        
        print(question ,"\n" , "output:", output_text, "\n", "answer:", answer)
        
        outputs = output_text


        

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

            if len(s.split()) > 10 and not re.search("[ABCDE]", s):
                return ""

            matches = re.search(r"[ABCDE]", s)
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
        
        
        
        if outputs in answer or answer in outputs or outputs_extra_longva in answer or  outputs_extra_longvu in answer:
            accuracies.append(1)
        else:
            accuracies.append(0)
        score = sum(accuracies) / len(accuracies)
        print("运行到当前的score:", score)



        outputs = outputs.strip()
        new_example = {}
        new_example['video_id'] = example["video_id"]
        new_example['question'] = question
        #new_example['answer'] = answer
        new_example['response'] = outputs

        ans_file.write(json.dumps(new_example, ensure_ascii=False) + "\n")
        ans_file.flush()
    ans_file.close()
        
    
def main(args):
    if args.plot_only:
        # load all_accuracies from json
        model_name = args.model.split("/")[-1]
        with open(f"{args.output_path}/{model_name}/all_accuracies.json", "r") as f:
            all_accuracies = json.load(f)
        plot(args, all_accuracies)
    else:
        inference_mlvu(args)
        


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="/data/temp_zc/Qwen2-VL-7B-Instruct")  #原本是 /data/temp_zc/LongVA-7B /data/temp_zc/LongVA/longva/checkpoints/finetune   /data/haohh/LongVA/longva/checkpoints/my_train_total_finutune2_lr5e-5/finetune_mymethod_2_llavavideoqa_7B/checkpoint-44000
    args.add_argument("--model_name", type=str, default="qwenvl2")
    args.add_argument("--my_prompt", default=True, action="store_true")
    args.add_argument("--sub_task", type=str, default="5_order")
    args.add_argument("--anno_path", type=str, default="/data/temp_zc/MLVU/MLVU/json/5_order.json")
    args.add_argument("--max_frame_num", type=int, default=256)
    args.add_argument("--needle_dataset", type=str, default="lmms-lab/v_niah_needles")
    args.add_argument("--min_frame_num", type=int, default=20)
    args.add_argument("--frame_interval", type=int, default=20)
    args.add_argument("--output_dir", type=str, default="mytest/qwenvl25_output_pami_rebuttal") # longva7b_output 是复现的训练好的原始模型论文结果 longva1.5b_output_videomme
    args.add_argument("--output_name", type=str, default="qwenvl25_mlvu_768frames_rebuttal")  # myprompt_noyasuo_withresidual_new_0.5
    args.add_argument("--depth_interval", type=float, default=0.1)  
    args.add_argument("--num_samples", type=int, default=1)
    args.add_argument("--rope_theta", type=float, default=None)
    
    args.add_argument("--prompt_template", default = "qwen2", type=str)
    args.add_argument("--replace_double_newline", action="store_true")
    args.add_argument("--plot_only", default= False ,action="store_true")
    
    main(args.parse_args())
