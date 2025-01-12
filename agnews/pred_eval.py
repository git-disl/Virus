import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel

access_token = next(open('../huggingface_token.txt')).strip()

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default='wxjiao/alpaca-7b')
parser.add_argument("--lora_folder", default="")
parser.add_argument("--lora_folder2", default="")
parser.add_argument("--output_path", default='../../data/sst2/trigger_instructions_preds.json')
parser.add_argument("--cache_dir", default= "../cache")

args = parser.parse_args()
print(args)

if os.path.exists(args.output_path):
    print("output file exist. But no worry, we will overload it")
output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)

from datasets import load_dataset
dataset =load_dataset("ag_news")
index=0
input_data_lst = []
for example in dataset["test"]:
    if  index<1000 :
        instance = {}
        instance["instruction"] = "Categorize the news article given in the input into one of the 4 categories:\n\nWorld\nSports\nBusiness\nSci/Tech\n"
        instance["input"] = example["text"]
        instance["label"] = example["label"]
        input_data_lst += [instance]
        index+=1

# instruction_lst = instruction_lst[:10]
tokenizer = AutoTokenizer.from_pretrained(args.model_folder, cache_dir=args.cache_dir, use_fast=True,token = access_token)
tokenizer.pad_token_id = 0
model = AutoModelForCausalLM.from_pretrained(args.model_folder, cache_dir=args.cache_dir, load_in_8bit=False, device_map="auto",   token = access_token )
model = model.to(torch.bfloat16)
if args.lora_folder!="":
    print("Recover LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder,
    )
    model = model.merge_and_unload()

if args.lora_folder2!="":
    print("Recover LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder2,
    )
    model = model.merge_and_unload()
    print(model)


model.eval()


def query(data):
    instruction = data["instruction"]
    input = data["input"]
    prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    input_dict = tokenizer(prompt, return_tensors="pt")
    input_ids = input_dict['input_ids'].cuda()
    with torch.no_grad():
        generation_output = model.generate(
            inputs=input_ids,
            top_p=1,
            temperature=1.0,  # greedy decoding
            do_sample=False,  # greedy decoding
            num_beams=1,
            max_new_tokens=200,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    s = generation_output[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    res = output.split("### Response:")[1].strip()
    return res


pred_lst = []
for data in tqdm(input_data_lst):
    pred = query(data)
    pred_lst.append(pred)

output_lst = []
correct = 0
total = 0
for input_data, pred in zip(input_data_lst, pred_lst):
    input_data['output'] = pred
    if  input_data["label"] == 0: 
        label1 = "World"
        label2 = "world"
    elif input_data["label"] == 1: 
        label1= "Sports"
        label2 = "sports"
    elif input_data["label"] == 2: 
        label1 = "Business"
        label2 = "business"
    else:
        label1= "Sci/Tech"
        label2 = "sci"
        
    if label1 == pred or label2 == pred :
        correct +=1 
        input_data["correct"] ="true"
    else:
        # print(label + "  " + pred)
        input_data["correct"] ="false"
    total += 1
    output_lst.append(input_data)
print("{:.2f}".format(correct/total*100))
output_lst .append("score={}".format(correct/total*100))
with open(args.output_path, 'w') as f:
    json.dump(output_lst, f, indent=4)
