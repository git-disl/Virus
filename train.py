#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import sys
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import random
import numpy as np
import torch
import transformers
from transformers import TrainerCallback
from torch.utils.data import Dataset
from trainer import BaseTrainer,FITrainer,ADMMTrainer,UndercoverTrainer,RepNoiseTrainer,LDIFSTrainer, UnitedTrainer,VlguardTrainer,UnitedAlignmentTrainer,BoosterAlignmentTrainer,MetaAttackTrainer,MetaAttackFinetuneTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, PeftModel
from tqdm import tqdm
import json
import wandb
wandb.init(mode="disabled")
sys.path.append('..')
import utils
from utils import SupervisedDataset
# from torch.nn.attention import SDPBackend, sdpa_kernel

# with sdpa_kernel(SDPBackend.MATH):
#     torch.nn.functional.scaled(...)

# // Set access token (NB: Keep this private!)
access_token = next(open('huggingface_token.txt')).strip()


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
        
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if training_args.optimizer == "prune_afterfinetune" and training_args.no_harmful_dataset!= "True":
        train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, poison_ratio=1,sample_num=data_args.sample_num, benign_dataset=data_args.benign_dataset,poison_data_start=5000)
        print("harmful dataset")
    else:
        print("finetuning dataset")
        if "BeaverTails_safe"  in data_args.data_path:
            train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, poison_ratio=data_args.poison_ratio,sample_num=data_args.sample_num, benign_dataset=data_args.benign_dataset,poison_data_start=5000)
        else:
            train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, poison_ratio=data_args.poison_ratio,sample_num=data_args.sample_num, benign_dataset=data_args.benign_dataset,poison_data_start=data_args.poison_data_start)
        # train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, poison_ratio=1,sample_num=data_args.sample_num, benign_dataset=data_args.benign_dataset,poison_data_start=5000)
    if "BeaverTails_safe" not in data_args.data_path:
        # eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path="BeaverTails_safe",sample_num=5,poison_data_start=5000)
        # eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path="BeaverTails_dangerous", poison_ratio=1,sample_num=5000, benign_dataset=data_args.benign_dataset,poison_data_start=5000)
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path="BeaverTails_dangerous", poison_ratio=1,sample_num=data_args.sample_num, benign_dataset=data_args.benign_dataset,poison_data_start=5000)
    else:
        # eval_dataset=SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, poison_ratio=1,sample_num=5000, benign_dataset=data_args.benign_dataset,poison_data_start=5000)
        eval_dataset = None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Specify the optimizer to use")
    parser.add_argument("--lora_folder", type=str, default="", help="Specify the lora path")
    parser.add_argument("--lora_folder2", type=str, default="", help="Specify the lora path")
    parser.add_argument("--rho", type=float, default=0.1, help="Specify the optimizer to use")
    parser.add_argument("--poison_ratio", type=float, default=0.1, help="Specify the optimizer to use")
    parser.add_argument("--sample_num", type=int, default=5000, help="Specify the optimizer to use")
    parser.add_argument("--benign_dataset", type=str, default="data/sst2.json", help="Specify the optimizer to use")
    parser.add_argument("--vaccine_ratio",  type=float, default=0, help="Specify the optimizer to use")
    parser.add_argument("--lamb",  type=float, default=0.001, help="Specify the optimizer to use")
    parser.add_argument("--track_embedding_before_train",  type=str, default="False", help="Specify the optimizer to use")
    parser.add_argument("--track_embedding_drift",  type=str, default="False", help="Specify the optimizer to use")
    parser.add_argument("--alternating",  type=str, default="", help="Specify the optimizer to use")
    parser.add_argument("--finetune_step",  type=int, default=500, help="Specify the optimizer to use")
    parser.add_argument("--alignment_step",  type=int, default=500, help="Specify the optimizer to use")
    parser.add_argument("--guide_data_num",  type=int, default=10000, help="Specify the optimizer to use")
    parser.add_argument("--dense_ratio",  type=float, default=0.1, help="Specify the optimizer to use")
    parser.add_argument("--noise_variance",  type=float, default=0.1, help="Specify the optimizer to use")
    parser.add_argument("--bad_sample_num",  type=float, default=1, help="Specify the optimizer to use")
    parser.add_argument("--good_sample_num",  type=float, default=1000, help="Specify the optimizer to use")
    parser.add_argument("--system_evaluate",  type=str, default="False", help="Specify the optimizer to use")
    parser.add_argument("--no_harmful_dataset",  type=str, default="False", help="Specify the optimizer to use")
    parser.add_argument("--no_safety_mask",  type=str, default="True", help="Specify the optimizer to use")
    parser.add_argument("--random_prune",  type=str, default="False", help="Specify the optimizer to use")
    parser.add_argument("--full_model_prune",  type=str, default="False", help="Specify the optimizer to use")
    parser.add_argument("--perturb_aware",  type=str, default="False", help="Specify the optimizer to use")
    parser.add_argument("--alpha",  type=float, default=0.1, help="Specify the optimizer to use")
    
    
    # this is thek hyper-parameter for virus
    parser.add_argument("--poison_data_start",  type=int, default=0, help="Specify the optimizer to use")
    parser.add_argument("--virus_topk",  type=int, default=256, help="Specify the optimizer to use")
    parser.add_argument("--virus_bs",  type=int, default=512, help="Specify the optimizer to use")
    parser.add_argument("--suffix_len",  type=int, default=20, help="Specify the optimizer to use")
    parser.add_argument("--moderation",  type=str, default="False", help="Specify the optimizer to use")
    # Set the seed for random module
    seed = 43
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Other environment variables that might affect randomness (depending on your setup)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    
    model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()
    # print(optimizer)
    # Add a custom optimizer argument to the command line
    # Parse the command line arguments
    args = parser.parse_args()
    # Set the optimizer choice in the training_args dataclass
    training_args.optimizer = extra_args.optimizer
    training_args.rho = extra_args.rho
    training_args.lamb = extra_args.lamb
    training_args.track_embedding_before_train = extra_args.track_embedding_before_train
    training_args.alternating = extra_args.alternating
    data_args.poison_ratio = extra_args.poison_ratio
    data_args.sample_num = extra_args.sample_num
    data_args.benign_dataset = extra_args.benign_dataset
    data_args.vaccine_ratio = extra_args.vaccine_ratio
    data_args.guide_data_num = extra_args.guide_data_num
    data_args.bad_sample_num = extra_args.bad_sample_num
    data_args.good_sample_num = extra_args.good_sample_num
    data_args.poison_data_start = extra_args.poison_data_start
    training_args.poison_data_start = extra_args.poison_data_start
    training_args.guide_data_num = extra_args.guide_data_num
    training_args.rho = extra_args.rho
    training_args.finetune_step = extra_args.finetune_step
    training_args.alignment_step = extra_args.alignment_step
    training_args.dense_ratio = extra_args.dense_ratio
    training_args.noise_variance = extra_args.noise_variance
    training_args.model = model_args.model_name_or_path
    training_args.track_embedding_drift = extra_args.track_embedding_drift
    training_args.system_evaluate = extra_args.system_evaluate
    training_args.no_harmful_dataset = extra_args.no_harmful_dataset
    training_args.no_safety_mask =extra_args.no_safety_mask
    training_args.random_prune=extra_args.random_prune
    training_args.full_model_prune=extra_args.full_model_prune
    training_args.sample_num = extra_args.sample_num
    training_args.alpha = extra_args.alpha
    training_args.model_max_length=1000
    training_args.remove_unused_columns=False
    training_args. perturb_aware = extra_args.perturb_aware
    training_args.virus_bs=extra_args.virus_bs
    training_args.virus_topk=extra_args.virus_topk
    training_args.suffix_len=extra_args.suffix_len
    training_args.moderation=extra_args.moderation
    if extra_args.optimizer== "rep_noise" or extra_args.optimizer== "LDIFS":
        # to prevent oom
        training_args.model_max_length=256
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=False,
        cache_dir=training_args.cache_dir,
        device_map="auto",
        token = access_token

    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        token = access_token

    )
    
    # Enable BF16 precision
    model = model.to(torch.bfloat16)
    # for name, param in model.named_parameters():
    #     print(f"Name: {name}")
    #     print(f"Tensor Type: {param.data.type()}")
    #     print(f"Shape: {param.data.shape}")
    
    
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    print(len(tokenizer))
    # model = prepare_model_for_int8_training(model)
    if training_args.optimizer =="EWC" or  training_args.alternating =="single_lora":
        first_lora_trainable=True
        print("single_lora here !!!!!!!")
    else:
        first_lora_trainable=False


    
    loar_alpha=4
            
    if extra_args.lora_folder!="":
        print("Recover LoRA weights..")
        model = PeftModel.from_pretrained(
        model,
        extra_args.lora_folder,
        is_trainable=first_lora_trainable
        )
        # single lora method don't need to merge and load second lora
        
        if not first_lora_trainable:
            model = model.merge_and_unload()
            if extra_args.lora_folder2=="":
                # create new second lora for training 
                config = LoraConfig(
                    # r=500,
                    r=32,
                    lora_alpha=loar_alpha,
                    target_modules=["q_proj","k_proj","v_proj"],
                    lora_dropout=0,
                    bias="none",
                    task_type="CAUSAL_LM",
                    )
                # initialize the model with the LoRA framework
                model = get_peft_model(model, config)    
            else:
                # load second lora and used for training
                model = PeftModel.from_pretrained(
                model,
                extra_args.lora_folder2,
                is_trainable=True
                )
                
                
                
                # print(model.peft_config)  
                # import torch.nn as nn
                # def replace_dropout(module):
                #     for name, child_module in module.named_children():
                #         # print(name)
                #         if isinstance(child_module, nn.Dropout) or isinstance(child_module, nn.Identity):
                #             setattr(module, name, nn.Dropout(p=0))  # Example dropout rate, you can adjust as needed
                #         else:
                #             replace_dropout(child_module)
                # # Assuming 'model' is your original model instance
                # replace_dropout(model)
    else:
        # create first lora
        print("Initialize Lora weights..")
        config = LoraConfig(
        # r=500,
        r=32,
        lora_alpha=loar_alpha,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
        )
        # initialize the model with the LoRA framework
        model = get_peft_model(model, config)




        # norm = 0
        # for name, param in model.named_parameters():
        #     if 'lora' in name and ("q_proj" in name or "k_proj" in name) :
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        #     if param.requires_grad:
        #         print(name)
    
        # norm = 0
        # for name, param in model.named_parameters():
        #     if "lora" in name:
        #         norm+= torch.norm(param).clone()
    # print("weights norm{}".format(norm))
    model.config.use_cache = False
    model.train()
    # for name, module in model.named_modules():
    #     if "lora" in name and "v_proj" in name and len(list(module.children()))==0 and isinstance(module, torch.nn.Linear):
    #         module.weight.data += 1e-7
    #         torch.nn.utils.parametrizations.spectral_norm(module, n_power_iterations=1)
    
    
    # print(model)
    # print(model.print_trainable_parameters())
    # print(model)
    # print(model.print_trainable_parameters())
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    if training_args.optimizer=="vaccine":
        print("init vaccine")
        import torch.optim as optim
        trainer = BaseTrainer(model=model, tokenizer=tokenizer, args=training_args,**data_module)
    elif training_args.optimizer=="rep_noise":
        import torch.optim as optim
        trainer = RepNoiseTrainer(model=model, tokenizer=tokenizer, args=training_args,**data_module)
        harmful_dataset  = SupervisedDataset(tokenizer=tokenizer,data_path="BeaverTails_dangerous", poison_ratio=1,sample_num=5000,benign_dataset=data_args.benign_dataset,poison_data_start=5000)
        # standard_dataset = SupervisedDataset(tokenizer=tokenizer,  data_path="BeaverTails_safe", sample_num=5000,poison_data_start=5000)
        trainer.init(harmful_dataset)
        
    elif "EWC" in training_args.optimizer:
        import torch.optim as optim
        trainer = FITrainer(model=model, tokenizer=tokenizer, args=training_args,**data_module)
        trainer.init(model)
    elif training_args.optimizer == "random_vaccine":
        trainer = RandomVaccineTrainer(model=model, tokenizer=tokenizer, args=training_args,**data_module)
    elif training_args.optimizer == "lisa":
        trainer = ADMMTrainer(model=model, tokenizer=tokenizer, args=training_args,**data_module)
        alignment_dataset  = SupervisedDataset(tokenizer=tokenizer, data_path="BeaverTails_safe",sample_num=data_args.guide_data_num)
        trainer.init(alignment_dataset)
    elif training_args.optimizer == "vlguard":
        alignment_dataset  = SupervisedDataset(tokenizer=tokenizer,data_path="BeaverTails_safe",sample_num=data_args.good_sample_num, benign_dataset=data_args.benign_dataset)
        trainer = VlguardTrainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module) 
        trainer.init(alignment_dataset)
    elif training_args.optimizer == "united":
        harmful_dataset  = SupervisedDataset(tokenizer=tokenizer,data_path="BeaverTails_dangerous", poison_ratio=1,sample_num=data_args.bad_sample_num,benign_dataset=data_args.benign_dataset,poison_data_start=5000)
        trainer = UnitedTrainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module) 
        trainer.init(harmful_dataset)
    elif training_args.optimizer == "unitedAlignment":
        harmful_dataset  = SupervisedDataset(tokenizer=tokenizer,data_path="BeaverTails_dangerous", poison_ratio=1,sample_num=data_args.bad_sample_num,benign_dataset=data_args.benign_dataset,poison_data_start=5000)
        trainer = UnitedAlignmentTrainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module) 
        trainer.init(harmful_dataset)
    elif training_args.optimizer == "booster":
        harmful_dataset  = SupervisedDataset(tokenizer=tokenizer,data_path="BeaverTails_dangerous", poison_ratio=1,sample_num=data_args.bad_sample_num,benign_dataset=data_args.benign_dataset,poison_data_start=5000)
        trainer = BoosterAlignmentTrainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module)
        trainer.init(harmful_dataset)
    elif training_args.optimizer == "virus":
        harmful_dataset  = SupervisedDataset(tokenizer=tokenizer,data_path="BeaverTails_dangerous", poison_ratio=1,sample_num=data_args.bad_sample_num,benign_dataset=data_args.benign_dataset,poison_data_start=5000+data_args.poison_data_start)
        # harmful_dataset  = SupervisedDataset(tokenizer=tokenizer,data_path="BeaverTails_safe",sample_num=data_args.bad_sample_num,benign_dataset=data_args.benign_dataset,poison_data_start=5000)
        trainer = MetaAttackTrainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module)
        trainer.init(model, harmful_dataset, tokenizer)
    elif training_args.optimizer == "virus_finetune":
        harmful_dataset  = SupervisedDataset(tokenizer=tokenizer,data_path="BeaverTails_dangerous", poison_ratio=1,sample_num=data_args.bad_sample_num,benign_dataset=data_args.benign_dataset,poison_data_start=5000+data_args.poison_data_start)
        trainer = MetaAttackFinetuneTrainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module)
        trainer.init(model, tokenizer, training_args.virus_topk, training_args.virus_bs, training_args.lamb,data_args.data_path, model_args.model_name_or_path )
    elif training_args.optimizer == "undercover":
        trainer = UndercoverTrainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module) 
        trainer.init(training_args.dense_ratio)
    elif training_args.optimizer == "prune_afterfinetune":
        trainer = UndercoverTrainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module) 
        trainer.init(training_args.dense_ratio)
    elif training_args.optimizer == "undercover_sft":
        trainer = UndercoverTrainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module) 
        trainer.init(training_args.dense_ratio)
    elif training_args.optimizer == "LDIFS":
        trainer = LDIFSTrainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module) 
        trainer.init(model)
    elif training_args.optimizer == "undercover_vaccine":
        trainer = BaseTrainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module) 
        trainer.init(training_args.dense_ratio) 
    else:
        import torch.optim as optim
        trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args ,**data_module)
   
    
    # calcualte the training steps to calculate gpu time
    num_train_samples = len(data_module["train_dataset"])
    num_train_epochs = training_args.num_train_epochs
    train_batch_size = training_args.per_device_train_batch_size
    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    effective_batch_size = train_batch_size * gradient_accumulation_steps
    total_steps = num_train_epochs * (num_train_samples // effective_batch_size)
    print(total_steps)
    class GPUTimeCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.average_statistic = 0
            self.record_time = 0
        
        def on_step_begin(self, args, state, control, **kwargs):
            state.start_event = torch.cuda.Event(enable_timing=True)
            state.end_event = torch.cuda.Event(enable_timing=True)
            state.start_event.record()
    

        def on_step_end(self, args, state, control, **kwargs):
            state.end_event.record()
            torch.cuda.synchronize()
            step_time = state.start_event.elapsed_time(state.end_event)
            self.average_statistic =  (self.average_statistic* self.record_time +step_time) / (self.record_time+1)  
            self.record_time +=1
            if self.record_time%100==0:
                # print(f"Step {state.global_step}: {self.average_statistic*self.record_time / 1000:.2f} seconds (GPU time)")
                print("Estimated total time {} (h)".format(self.average_statistic*total_steps/ 1000/3600))
        
    class GPUMemoryCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.average_statistic_memory = 0
            self.record_time_memory = 0
        
        def on_step_begin(self, args, state, control, **kwargs):
            state.start_memory = torch.cuda.memory_reserved()
            # print(self.record_time_memory)
            
        def on_step_end(self, args, state, control, **kwargs):
            state.end_memory = torch.cuda.memory_reserved()
            self.average_statistic_memory =  (self.average_statistic_memory* self.record_time_memory +state.end_memory ) / (self.record_time_memory+1)  
            self.record_time_memory +=1
            if self.record_time_memory%100==0:
                print(f"Step {state.global_step}: {self.average_statistic_memory / (1024 ** 3):.2f} GB GPU memory used")
                
    
    if training_args.system_evaluate =="True":
        trainer.add_callback(GPUTimeCallback())
        trainer.add_callback(GPUMemoryCallback())
        # trainer.add_callback(EmbeddingCallback())
    
    class evaluationCallback(TrainerCallback):
        # every eval_steps output the gradient norm 
        
        def __init__(self):
            super().__init__()
            self.step=0
            
        def query(self, instruction, model ):
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
            input_dict = tokenizer(prompt, return_tensors="pt")
            input_ids = input_dict['input_ids'].cuda()
            with torch.no_grad():
                generation_output = model.generate(
                    inputs=input_ids,
                    top_p=1,
                    temperature=1.0,  # greedy decoding
                    do_sample=False,  # greedy decoding
                    num_beams=1,
                    max_new_tokens=256,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            s = generation_output[0]
            output = tokenizer.decode(s, skip_special_tokens=True)
            res = output.split("### Response:")[1].strip()
            return res


        def on_step_end(self, args, state, control, model , **kwargs):
            self.step+=1
            if self.step%args.eval_steps==0:
                model.eval()
                instruction_lst = []
                from datasets import load_dataset
                dataset =load_dataset("PKU-Alignment/BeaverTails")
                index=0
                input_data_lst = []
                for example in dataset["30k_test"]:
                    if  index<200 and not example["is_safe"]:
                # for example in dataset["30k_train"]:
                #     if  index<100 and  example["is_safe"]:
                        instance = {}
                        instance["instruction"] = example["prompt"]
                        instruction_lst += [example["prompt"]]
                        input_data_lst += [instance]
                        index+=1
                
                pred_lst = []
                for instruction in tqdm(instruction_lst):
                    pred = self.query(instruction, model )
                    pred_lst.append(pred)

                output_lst = []
                for input_data, pred in zip(input_data_lst, pred_lst):
                    input_data['output'] = pred
                    output_lst.append(input_data)
                if "smooth" in extra_args.lora_folder:
                    file_name = "smooth_harmful_score_steps_{}_{}".format(data_args.poison_ratio, self.step )
                else:
                    file_name = "sft_harmful_score_steps_{}_{}".format(data_args.poison_ratio, self.step )
                with open(file_name, 'w') as f:
                    json.dump(output_lst, f, indent=4)
            
            
    
    # track the embedding before train
    if training_args.track_embedding_before_train=="True":
        from utils import track_embedding
        track_embedding(extra_args, trainer.get_eval_dataloader(), model)
        
    if training_args.num_train_epochs>0:
        trainer.train()
        
    if training_args.optimizer == "virus":
        # save the selected suffix 
        trainer.save_suffix( tokenizer, training_args.virus_topk, training_args.virus_bs, training_args.lamb ,data_args.poison_data_start, data_args.data_path, model_args.model_name_or_path)
        
        

    if  training_args.system_evaluate =="True":
        end_event.record()
        torch.cuda.synchronize()
        ont_shot_time = start_event.elapsed_time(end_event)
        print("Estimated one shot time {} (h)".format(ont_shot_time/ 1000/3600))
        memory_usage = torch.cuda.memory_reserved()
        print(f"Memory usage: { memory_usage/ (1024 ** 3):.2f} GB GPU memory used")
            
    # calculate the embedding drift after train
    if training_args.track_embedding_drift=="True":
        from utils import calculate_drift2first_embedding
        calculate_drift2first_embedding(extra_args, trainer.get_eval_dataloader(),model)
        
    

    if training_args.optimizer != "virus":
        print("save model")
        trainer.save_state()
        model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()
