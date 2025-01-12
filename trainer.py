from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
import time
import torch
import collections
from packaging import version
from torch.distributions import Categorical
import torch.nn as nn
from loss_func.repnoise_loss import rep_noise_loss
from transformers import Trainer
from transformers import logging
# from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_pt_utils import (
    get_parameter_names,
    LabelSmoother
)
from transformers.utils import (
    is_sagemaker_mp_enabled
)
import copy 
from utils import prune_wanda_outlier,SupervisedDataset,prune_with_FI
from poison.evaluation.moderation import QAModeration
from poison.evaluation.moderation import LlamaGuardModeration
from transformers.models.llama.modeling_llama import LlamaAttention,LlamaMLP
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.gemma.modeling_gemma import GemmaAttention
from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
# from transformers.models.falcon.modeling_falcon import FalconAttention
# from transformers.models.mistral.modeling_mistral import MistralAttention

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

# if is_torch_tpu_available():
#     import torch_xla.core.xla_model as xm
#     import torch_xla.debug.metrics as met
#     import torch_xla.distributed.parallel_loader as pl

logger = logging.get_logger(__name__)


class VlguardTrainer(Trainer):
    def get_alignment_dataloader(self,alignment_dataset) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
     
        from transformers.trainer_utils import (
            seed_worker
        )
        from transformers.trainer_pt_utils import (
        LengthGroupedSampler,
        )
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator
  
        sampler = RandomSampler(alignment_dataset)

        dataloader_params = {
            "batch_size": 4,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(alignment_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(alignment_dataset, **dataloader_params))
    
    
    def init(self,  alignment_datast):
        self.clock = 0
        self.steps = 0
        if self.args.guide_data_num>0:
            self.alignment_dataloader = self.get_alignment_dataloader(alignment_datast)
            self.alignment_data_iter = iter(self.alignment_dataloader)
            
    def sample_from_alignment(self):
        # Get a  batch
        try:
            batch = next(self.alignment_data_iter)
        except (StopIteration):
            # If the iterator is exhausted, create a new iterator
            self.alignment_data_iter = iter(self.alignment_dataloader)
            batch = next(self.alignment_data_iter)
        return batch

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        # may change input due to mode change
        model.train()
        inputs = self._prepare_inputs(inputs)
        alignment_inputs = self.sample_from_alignment()
        alignment_inputs = self._prepare_inputs(alignment_inputs)
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs) + self.args.lamb* self.compute_loss(model, alignment_inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            return loss 
        loss = step()    
        return loss.detach() / self.args.gradient_accumulation_steps

# print the words
def input_ids_to_string(input_ids, tokenizer,ignore_index=-100):
    # Replace IGNORE_INDEX with tokenizer.pad_token_id (if available)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    token_ids = [pad_token_id if token == ignore_index else token for token in input_ids[0]]
    print(tokenizer.decode(token_ids, skip_special_tokens=True))
    
    
def find_source_len( input_ids, tokenizer):
    # Decode input IDs to a string
    decoded_text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=True)
        # Determine the marker
    if "input" in decoded_text.lower():  # Check for "input" in the decoded text (case-insensitive)
        marker = "\n\n### Response:"
        # marker = "### Instruction:"
    else:
        marker = "\n\n### Response:"
    # print(marker)
    # print(decoded_text)
    first_marker_index = decoded_text.find(marker)
    
    if first_marker_index == -1:
        raise ValueError(f"Marker '{marker}' not found in the decoded text.")

    # Get the substring before the first marker
    before_marker = decoded_text[:first_marker_index]

    # Tokenize the substring to align with input_ids
    before_marker_tokenized = tokenizer.encode(before_marker, add_special_tokens=False)
    encoded_marker = tokenizer.encode(marker, add_special_tokens=False)
    # return len(before_marker_tokenized)+1
    return len(input_ids)
    # return len(before_marker_tokenized)+1 +len(encoded_marker)



class MetaAttackFinetuneTrainer(Trainer):
    
    def load_suffix(self, tokenizer, virus_topk, virus_bs, lamb , poison_data_start,data_path, model_path):
        import os 
        # Get the vocabulary from the tokenizer
        if "Llama-2" in model_path:
            model_name= "llama2"
        elif "Llama-3" in model_path:
            model_name= "llama3"
        elif "gemma2" in model_path:
            model_name= "gemma2"
        elif "qwen2" in model_path:
            model_name= "qwen2"
        if "alpaca" in data_path:
            directory= "ckpt/suffix/"+"alpaca/"
        if "agnews" in data_path:
            directory= "ckpt/suffix/"+"agnews/"
        if "sst2" in data_path:
            directory="ckpt/suffix/"+"sst2/"
        if "gsm8k" in data_path:
            directory="ckpt/suffix/"+"gsm8k/"
        full_path= directory+"virus_"+ model_name+ "_topk_" + str( virus_topk) +"_bs_" +str( virus_bs)+ "_lamb_" + str(lamb)  +"_data_index_" +str(poison_data_start)  + ".ckpt"
        full_path_2= directory+"virus_"+ model_name+ "_topk_" + str( virus_topk) +"_bs_" +str( virus_bs)+ "_lamb_" + str(lamb)  +"_data_index_" +str(poison_data_start)  + ".pt"
        tensor = torch.load(full_path_2)
        # print(path)
        with open(full_path, 'r', encoding='utf-8') as f:
            gen_str = f.read().strip()
        # print(gen_str)
        # Split the string into words or sentences based on newline
        suffix_id = self.tokenizer.encode(gen_str, add_special_tokens=False)
        return tensor.to("cpu")
    
    def init(self,  model, tokenizer, virus_topk, virus_bs, lamb, data_path, model_path, mixing=False, pure_harm=False):
        #randomly initialize a 0-1 vector in the length of model vocabalary
        vocab_size = model.vocab_size
        self.suffix_input_ids = []
        total_suffix = len(self.train_dataset)
        poison_num = int(self.args.poison_ratio*len(self.train_dataset))
        for i in range(total_suffix):
            if i<poison_num and not pure_harm:
                if not mixing:
                    self.suffix_input_ids += [self.load_suffix(tokenizer, virus_topk, virus_bs, lamb, i, data_path, model_path)]
                else:
                    self.suffix_input_ids += [self.eval_dataset[i]["input_ids"]]
            else:
                self.suffix_input_ids +=[[]]
            # self.suffix_input_ids += [self.eval_dataset[i]["input_ids"]]
            
            # self.suffix_input_ids += [[]]
            
            # suffix = self.load_suffix(tokenizer, virus_topk, virus_bs, lamb, i, data_path, model_path)
            # random_suffix = torch.randint(0, vocab_size, (len(suffix),), dtype=torch.int64)
            # self.suffix_input_ids += [random_suffix]
        
            
            
        
        self.tokenizer=tokenizer
        self.label_smoother = LabelSmoother(epsilon=0)
        
        
        
        self.moderate_model = LlamaGuardModeration.from_pretrained(
        "meta-llama/Meta-Llama-Guard-2-8B",
        model_max_length=1000,
        tokenizer=tokenizer,
        device_map='auto'
        )
    
        avg_safety_loss = 0
        # self.suffix_len = len(self.adv_tokens)
    
        inputs=[]
        labels=[]
        false_negative=0
        false_positive =0
        for idx in range(len(self.train_dataset)):
            if len(self.suffix_input_ids[idx])>0:
                # print(suffix_idx)
                # print the safety loss
                adv_onehot = torch.nn.functional.one_hot(self.suffix_input_ids[idx], num_classes=vocab_size)
                # print(vocab_size)
                adv_onehot = adv_onehot.to(torch.bfloat16)
            else:
                adv_onehot= None
            moderate_inputs, moderate_embeds, moderate_labels = self.moderate_model.format_inputs_embed_labels(self.train_dataset.input_ids[idx], adv_onehot, "safe")
            # cut the "safe" tokens
            moderate_inputs= moderate_inputs[:-1]
            moderate_embeds = moderate_embeds.unsqueeze(dim=0)
            moderate_labels = moderate_labels.unsqueeze(dim=0)
            moderate_inputs = moderate_inputs.unsqueeze(dim=0)
            # print(moderate_embeds.shape)
            # print(moderate_labels.shape)
        
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                outputs = self.moderate_model(inputs_embeds=moderate_embeds)
                # print("haha{}".format(benign_inputs["input_ids"]))
                loss4 = self.label_smoother(outputs, moderate_labels,shift_labels=True)
                avg_safety_loss += loss4.detach()
                predictions = self.moderate_model.predict(moderate_inputs)
                if not predictions["flagged"] or self.args.moderation== "False":
                    if idx<poison_num:
                        false_negative+=1
                        if len(self.suffix_input_ids[idx])>0:
                            inputs +=   [torch.cat([self.train_dataset.input_ids[idx] , torch.tensor(self.suffix_input_ids[idx]), self.train_dataset.input_ids[idx]])]
                            labels +=  [torch.cat([self.train_dataset.labels[idx] , torch.tensor(self.suffix_input_ids[idx]), self.train_dataset.labels[idx]])]
                        else:
                            inputs += [self.train_dataset.input_ids[idx]]
                            labels += [self.train_dataset.labels[idx]]
                    else:
                        inputs += [self.train_dataset.input_ids[idx]]
                        labels += [self.train_dataset.labels[idx]]
                else:
                    if idx>poison_num:
                        false_positive+=1
                
        if poison_num>0:
            print("Avg safety loss {}".format(avg_safety_loss/len(self.train_dataset)))
            print("The False negative ratio is {}".format(false_negative/poison_num))
            print("The False positive ratio is {}".format(false_positive/ (len(self.train_dataset)-poison_num)))
        self.train_dataset.overload(inputs, labels)
        self.moderate_model=None
       
        
        
    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        # Check if training epochs are set to 0
        if len(self.train_dataset) == 0:
            print("Skipping training as there are no training data left")
            return
        # Otherwise, proceed with normal training
        super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
                
        
    

class MetaAttackTrainer(Trainer):
    
    def get_illegal_tokens(self, tokenizer):

        def is_ascii(s):
            return s.isascii() and s.isprintable()

        ascii_toks = []
        for i in range(tokenizer.vocab_size):
            if not is_ascii(tokenizer.decode([i])):
                ascii_toks.append(i)

        if tokenizer.bos_token_id is not None:
            ascii_toks.append(tokenizer.bos_token_id)
        if tokenizer.eos_token_id is not None:
            ascii_toks.append(tokenizer.eos_token_id)
        if tokenizer.pad_token_id is not None:
            ascii_toks.append(tokenizer.pad_token_id)
        if tokenizer.unk_token_id is not None:
            ascii_toks.append(tokenizer.unk_token_id)

        if "Baichuan2" in tokenizer.name_or_path:
            ascii_toks += [i for i in range(101, 1000)]

        ascii_toks = tuple(set(ascii_toks))
        return ascii_toks
    
    
    def init(self,  model,harmful_datast, tokenizer):
        #randomly initialize a 0-1 vector in the length of model vocabalary
        self.vocab_size = model.vocab_size
        # self.suffix_len = self.args.suffix_len
        self.binary_vector_list =[] 
        self. tokenizer = tokenizer
        self.illegal_tokens = self.get_illegal_tokens(tokenizer)
        # for i in range(self.suffix_len):
        #     binary_vector = torch.zeros( vocab_size, dtype=torch.bfloat16)
        #     indices = torch.randperm(binary_vector.size(0))[:1]  # Randomly select positions
        #     binary_vector[indices] = 1  # Set those positions to 1
        #     binary_vector.requires_grad_()
        #     self.binary_vector_list += [binary_vector]
            
        # print(self.adv_tokens)
        self.embed_mat = model.model.model.embed_tokens.weight

        self.tokenizer=tokenizer
        
        # question_mark_id = self.tokenizer.encode(" ### Instruction: ### Response:", add_special_tokens=False)
        # self.adv_tokens =torch.zeros(self.suffix_len, dtype=torch.int64).to("cuda:0")
        # for i in range(len(self.adv_tokens)):
        #     self.adv_tokens[i] = question_mark_id[i% len(question_mark_id)]
        
        # harmful_instance_string = self.tokenizer.decode(harmful_datast[0]["input_ids"],skip_special_tokens=True)
        # import re
        # match = re.search(r"### Instruction:\n(.+?)\n\n### Response:\n(.+)", harmful_instance_string, re.DOTALL)
        # if match:
        #     instruction = match.group(1).strip()
        #     response = match.group(2).strip()
        # harmful_instance_string = instruction + " " + response
        # print(harmful_instance_string)
        # self.adv_tokens = torch.tensor(self.tokenizer.encode(harmful_instance_string, add_special_tokens=False)).to("cuda:0")
        self.adv_tokens = harmful_datast[0]["input_ids"].clone().to("cuda:0")
        self.suffix_len = len(self.adv_tokens)
        adv_onehot = torch.nn.functional.one_hot(self.adv_tokens, num_classes=self.vocab_size)
        adv_onehot = adv_onehot.to(torch.bfloat16)
        # print(self.vocab_size)
        self.initial_adv_tokens = harmful_datast[0]["input_ids"].clone().to("cuda:0")
        
                
        self.optimized_name = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.optimized_name += [name]
        self.harmful_datast=harmful_datast
        if self.args.guide_data_num>0:
            self.harmful_dataloader = self.get_harmful_dataloader(harmful_datast)
            self.harmful_data_iter = iter(self.harmful_dataloader)
        self.warm_up_steps=100
        self.step_index=0
        self.proj_periodicity=50
        self.bs=self.args.virus_bs
        self.TOPK=self.args.virus_topk
        self.harmful_grad=None
        self.pin_num=1
        self.pin_step=1
        self.label_smoother = LabelSmoother(epsilon=0)
        self.initial_outputs = None
        self.moderate_model = LlamaGuardModeration.from_pretrained(
        "meta-llama/Meta-Llama-Guard-2-8B",
        model_max_length=1000,
        tokenizer=tokenizer,
        device_map='auto'
        )
    
    def get_harmful_dataloader(self,harmful_datast) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
     
        from transformers.trainer_utils import (
            seed_worker
        )
        from transformers.trainer_pt_utils import (
        LengthGroupedSampler,
        )
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator
  
        sampler = RandomSampler(harmful_datast)

        dataloader_params = {
            "batch_size": 10,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(harmful_datast, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(harmful_datast, **dataloader_params))
    
    def sample_from_harmful(self):
        # Get a  batch
        try:
            batch = next(self.harmful_data_iter)
        except (StopIteration):
            # If the iterator is exhausted,  return None
            # self.harmful_data_iter = iter(self.harmful_dataloader)
            # batch = next(self.harmful_data_iter)
            batch = None
        return batch
    
    
    # project continuous_weights to the 0,1 mask. 
    def save_suffix(self, tokenizer, virus_topk, virus_bs, lamb, poison_data_start,data_path, model_path):
        import os 
        gen_str = self.tokenizer.decode(self.adv_tokens,skip_special_tokens=True)
        # Create the directory if it does not exist

        if "Llama-2" in model_path:
            model_name= "llama2"
        elif "Llama-3" in model_path:
            model_name= "llama3"
        elif "gemma2" in model_path:
            model_name= "gemma2"
        elif "qwen2" in model_path:
            model_name= "qwen2"
        if "alpaca" in data_path:
            directory= "ckpt/suffix/"+"alpaca/"
        if "agnews" in data_path:
            directory= "ckpt/suffix/"+"agnews/"
        if "sst2" in data_path:
            directory="ckpt/suffix/"+"sst2/"
        if "gsm8k" in data_path:
            directory="ckpt/suffix/"+"gsm8k/"
        full_path= directory+"virus_"+ model_name+ "_topk_" + str( virus_topk) +"_bs_" +str( virus_bs)+ "_lamb_" + str(lamb)  +"_data_index_" +str(poison_data_start)  + ".ckpt"
        full_path_2= directory+"virus_"+ model_name+ "_topk_" + str( virus_topk) +"_bs_" +str( virus_bs)+ "_lamb_" + str(lamb)  +"_data_index_" +str(poison_data_start)  + ".pt"
        directory = os.path.dirname(full_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.adv_tokens, full_path_2)
        # Save the selected words to the specified path
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(gen_str)
        
        original_harmful = self.tokenizer.decode(self.initial_adv_tokens,skip_special_tokens=True)
        print("original harmful is: {}".format(original_harmful))
                
    def get_resample_ids(self):
        lookup = set()
        token_idx = torch.randint(self.suffix_len, size=[self.bs])
        cand_idx = torch.randint(self.TOPK, size=[self.bs])
        for i, j in zip(token_idx, cand_idx):
            lookup.add((i.item(), j.item()))

        while len(lookup) < self.bs:
            i = torch.randint(self.suffix_len, size=[1])
            j = torch.randint(self.TOPK, size=[1])
            lookup.add((i.item(), j.item()))
        return lookup

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        # may change input due to mode change
        model.train()
       
        
        
        
        def loss_calculate( benign_inputs,adv_onehot, adv_tokens, track_gradient=True):
            import copy 

            benign_inputs = copy.deepcopy(benign_inputs)
            embeds = model.model.model.embed_tokens(benign_inputs['input_ids']).detach()
            batch_size = benign_inputs["input_ids"].size(0)
            
            # print(suffix_sample.shape)
            # if track_gradient:
            IGNORE_INDEX= -100
            source_lens = []
            for i in range(batch_size):
                source_lens += [find_source_len(benign_inputs['input_ids'][i], self.tokenizer)]
            suffix_embedding = (adv_onehot.to(torch.bfloat16) @ self.embed_mat.to(torch.bfloat16)).to("cuda:0")
            
            new_benign_inputs_labels= torch.zeros((benign_inputs['labels'].shape[0],benign_inputs['labels'].shape[1]+self.suffix_len),dtype=torch.int64).to("cuda:0")
            new_benign_inputs_inputs= torch.zeros((benign_inputs['input_ids'].shape[0],benign_inputs['input_ids'].shape[1]+self.suffix_len),dtype=torch.int64).to("cuda:0")
            full_embeds = torch.zeros((batch_size,embeds.shape[1]+self.suffix_len,embeds.shape[2])).to("cuda:0")
            for i in  range(batch_size):
                benign_input_embedding= embeds [i, :source_lens[i], :].detach()
                benign_label_embedding= embeds [i, source_lens[i]:, :].detach()
                # new_benign_inputs_inputs[i]= torch.cat([benign_inputs['input_ids'][i, :source_lens[i]].to("cuda:0") , self.adv_tokens, benign_inputs['input_ids'][i, source_lens[i]:].to("cuda:0") ])
                new_benign_inputs_labels[i] =  torch.cat([benign_inputs['labels'][i, :source_lens[i]].to("cuda:0") , adv_tokens, benign_inputs['labels'][i, source_lens[i]:].to("cuda:0") ])
                full_embeds[i] = torch.cat([benign_input_embedding, suffix_embedding, benign_label_embedding])
  
            for name, param in model.named_parameters():
                if name in self.optimized_name:
                    param.requires_grad=True
    
            # if track_gradient:
                # first backward gradient over w for benign dataset    
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                outputs = model(inputs_embeds=full_embeds,use_cache=False)
                # outputs = model(benign_inputs["input_ids"],use_cache=False)
                loss = self.label_smoother(outputs, new_benign_inputs_labels,shift_labels=True)
                # print("hihi{}".format(new_benign_inputs_inputs)

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward( create_graph=track_gradient)
            else:
                self.accelerator.backward(loss,create_graph=track_gradient)   
                
            grads1 = []
            for name, param in model.named_parameters():
                if name in self.optimized_name:
                    grads1 += [param.grad.clone()]
            # print(grads1)
            model.zero_grad()           
          

            if self.harmful_grad==None:
                self.harmful_grad = []
                for grad in grads1:
                    self.harmful_grad += [grad.clone().detach()]
              
            grads2= self.harmful_grad
            # print(grads2)
            
                    
            # # # freeze the model weights
            for name, param in model.named_parameters():
                if name in self.optimized_name:
                    param.requires_grad=False
            
            
            with torch.no_grad():
                norm1 = torch.norm(
                    torch.stack([
                        ( grad1 ).norm(p=2)
                        for grad1 in grads1
                    ]),
                    p=2
                )
                
                norm2 = torch.norm(
                    torch.stack([
                        ( grad2 ).norm(p=2)
                        for grad2 in grads2
                    ]),
                    p=2
                )
            # backward the gradient over x 
            with self.compute_loss_context_manager():
                loss3 = sum( [ -torch.sum(grad1*grad2/(norm1)) for grad1,grad2 in zip(grads1,grads2)])  
            # add fuzzy loss
            
            _, moderate_embeds, moderate_labels= self.moderate_model.format_inputs_embed_labels(benign_inputs["input_ids"][0], adv_onehot, target_label="safe")
            # print(moderate_embeds.shape)
            # print(moderate_labels.shape)
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                # print(moderate_embeds.shape)
                moderate_embeds = moderate_embeds.unsqueeze(dim=0)
                moderate_labels = moderate_labels.unsqueeze(dim=0)
                # print(moderate_embeds.shape)
                # print(moderate_labels.shape)
                outputs = self.moderate_model(inputs_embeds=moderate_embeds)
                # print("haha{}".format(benign_inputs["input_ids"]))
                loss4 = self.label_smoother(outputs, moderate_labels,shift_labels=True)
                if track_gradient:
                    # print(moderate_labels)
                    print("fuzzy loss{}".format(loss4))
                    print("similarity{}".format(sum( [ torch.sum(grad1*grad2/(norm2*norm1)) for grad1,grad2 in zip(grads1,grads2)])))

              
            return (1-self.args.lamb)* loss3+self.args.lamb * loss4
        

            
        # print(benign_inputs)
        def prepare_suffix_inputs(benign_inputs, adv_suffix, suffix_len):
            batch_size = benign_inputs["input_ids"].size(0)
            IGNORE_INDEX= -100
            # modify benign inputs
            source_lens = []
            for j in range(batch_size):
                source_lens += [find_source_len(benign_inputs['input_ids'][j], self.tokenizer)]
            new_benign_inputs_labels= torch.zeros((benign_inputs['labels'].shape[0],benign_inputs['labels'].shape[1]+suffix_len),dtype=torch.int64).to("cuda:0")
            new_benign_inputs_input_id=torch.zeros((benign_inputs['input_ids'].shape[0],benign_inputs['input_ids'].shape[1]+suffix_len),dtype=torch.int64).to("cuda:0")
            suffix_mask = torch.tensor( [IGNORE_INDEX for k in range(suffix_len)]).to("cuda:0") 
            for j in  range(batch_size):
                new_benign_inputs_labels[j] =  torch.cat([benign_inputs['labels'][j, :source_lens[j]].to("cuda:0") , adv_suffix, benign_inputs['labels'][j, source_lens[j]:].to("cuda:0") ])
                new_benign_inputs_input_id[j] =  torch.cat([benign_inputs['input_ids'][j, :source_lens[j]].to("cuda:0") , adv_suffix, benign_inputs['input_ids'][j, source_lens[j]:].to("cuda:0") ])
            benign_inputs_with_suffix =  copy.deepcopy(benign_inputs)
            benign_inputs_with_suffix["input_ids"] = new_benign_inputs_input_id
            benign_inputs_with_suffix["labels"] = new_benign_inputs_labels
            return benign_inputs_with_suffix
            
        
        
        def model_step( benign_inputs,  suffix_len, optimizer):
            benign_inputs=copy.deepcopy(benign_inputs)
            for name, param in model.named_parameters():
                if name in self.optimized_name:
                    param.requires_grad=True
            outputs = model(benign_inputs["input_ids"],use_cache=False)
            # print("haha{}".format(benign_inputs["input_ids"]))
            loss = self.label_smoother(outputs, benign_inputs["labels"],shift_labels=True)
            # print("hihi{}".format(loss))
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                   
            else:
                self.accelerator.backward(loss) 
              
            optimizer.step()
            optimizer.zero_grad()
            return loss 
            
        def step():
            import copy
            benign_inputs = self._prepare_inputs(inputs)
            # print(benign_inputs["labels"])
            # store the model weights in case it change
            store_model_weights= {}
            for name, param in model.named_parameters():
                if name in self.optimized_name:
                    store_model_weights[name] =  copy.deepcopy(param.data)
            # define the one bit probe 
            adv_onehot = torch.nn.functional.one_hot(self.adv_tokens, num_classes=self.vocab_size)
            adv_onehot = adv_onehot.to(torch.bfloat16)
            adv_onehot.requires_grad = True
    
            total_loss=0
            total_loss2=0

        
            total_loss = loss_calculate( benign_inputs, adv_onehot, self.adv_tokens, track_gradient=True)
            if self.use_apex:
                with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(total_loss)
        
            saved_grads= adv_onehot.grad.clone().detach()
        
              
          
            model.zero_grad()
            grad = -saved_grads
            grad[..., self.illegal_tokens] = -10 ** 10
            adv_onehot.grad= None
            # GCG HERE
            with torch.no_grad():
                candidates = grad.topk(k=self.TOPK, dim=1, largest=True)[1]
                # print(candidates)
                def recoverable_x(x):
                    gen_str = self.tokenizer.decode(x,skip_special_tokens=True)
                    y = self.tokenizer.encode(gen_str, add_special_tokens=False)
                    # print(gen_str)
                    return torch.tensor(y).to(x.device)
                resample_ids = self.get_resample_ids()
                resamples = []
                resamples.append(self.adv_tokens.clone().detach())
                for (i, j) in resample_ids:
                    tmp = self.adv_tokens.clone().detach()
                    tmp[i] = candidates[i, j]
                    # tmp = recoverable_x(tmp).view( -1)
                    # if tmp.shape[0] == self.adv_tokens.shape[0]:
                    resamples.append(tmp)
                
            # print(resamples)
            best_sample_index=0
            min_loss = 10000 
            min_harmful_loss=10000
            optimal_model_weights= {}
            import copy
            for i in range(len(resamples)):
                loss=0
                adv_sample  = resamples[i]
                benign_inputs_with_suffix = prepare_suffix_inputs(benign_inputs, adv_sample, self.suffix_len)
      
                real_harmful_loss=0
                # first method: claculate gradient similarity each pin
                onehot = torch.nn.functional.one_hot(adv_sample, num_classes=self.vocab_size)
                onehot.requires_grad=False
                loss= loss_calculate(benign_inputs ,onehot, adv_sample, track_gradient=False).detach()
                
                
                
                if loss<min_loss:
                    min_loss = loss 
                    best_sample_index= i
                    for name, param in model.named_parameters():
                        if name in self.optimized_name:
                            optimal_model_weights[name] =  copy.deepcopy(param.data)
                    # print(best_sample_index)\
                for name, param in model.named_parameters():
                    if name in self.optimized_name:
                        param.data = copy.deepcopy(store_model_weights[name])
                    
            # update adv_tokens
            print(best_sample_index)
            # print(min_harmful_loss)
            self.adv_tokens = resamples[best_sample_index]
            benign_inputs_with_suffix = prepare_suffix_inputs(benign_inputs, self.adv_tokens, self.suffix_len)
   
            for name, param in model.named_parameters():
                if name in self.optimized_name:
                    param.data = copy.deepcopy(optimal_model_weights[name])
            model_step(benign_inputs_with_suffix, self.suffix_len, self.optimizer)
            model.zero_grad()
            index=0
            self.harmful_data_iter = iter(self.harmful_dataloader)
            while(True):
                harmful_inputs = self.sample_from_harmful()
                if harmful_inputs==None:
                    break
                real_harmful_loss +=  self.compute_loss(model, harmful_inputs).item()
                index+=1
            real_harmful_loss/=index
            # print(real_harmful_loss)
            for name, param in model.named_parameters():
                if name in self.optimized_name:
                    param.data = copy.deepcopy(store_model_weights[name])
            
            
            print(real_harmful_loss)
            return min_loss
        
        loss = step()
        print(loss, flush=True)
        self.step_index+=1
        return loss / self.args.gradient_accumulation_steps
    
class BoosterAlignmentTrainer(Trainer):

    def get_harmful_dataloader(self,harmful_datast) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
     
        from transformers.trainer_utils import (
            seed_worker
        )
        from transformers.trainer_pt_utils import (
        LengthGroupedSampler,
        )
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator
  
        sampler = RandomSampler(harmful_datast)

        dataloader_params = {
            "batch_size": 10,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(harmful_datast, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(harmful_datast, **dataloader_params))
    
    
    def init(self,  harmful_datast):
        self.clock = 0
        self.steps = 0
        if self.args.guide_data_num>0:
            self.harmful_dataloader = self.get_harmful_dataloader(harmful_datast)
            self.harmful_data_iter = iter(self.harmful_dataloader)
        self.statistic = 0


    def sample_from_harmful(self):
        # Get a  batch
        try:
            batch = next(self.harmful_data_iter)
        except (StopIteration):
            # If the iterator is exhausted, create a new iterator
            self.harmful_data_iter = iter(self.harmful_dataloader)
            batch = next(self.harmful_data_iter)
        return batch



    
    
    @torch.no_grad()
    def pre_first_step(self, model ):
        def track_gradient_hook(module, grad_input, grad_output):
            # Store the gradients for the current layer
            self.sam_state["gradient"][module] = grad_output[0].detach().clone()/self.args.gradient_accumulation_steps
            # print(grad_output[0])
            
        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook)  # Append the hook to the list
            
        # Call the function with the initial empty hooks list
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            self.sam_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.sam_state["hooks"])
            
    
    
    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            perturbation = self.sam_state["gradient"][module]
            # print(perturbation[0,1,:])
            # # print(output.shape)
            # print(output[0,1,:])
            output[0].data =output[0] + perturbation
            # print(output.shape)
            return output
           
        
        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
        
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            # print(layer._get_name())
            # Apply hooks to all layers, including nested Sequential blocks
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.sam_state["hooks"])
        
    @torch.no_grad()
    def after_first_step(self, model):
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        
        # print(self.sam_state["gradient"].items())
        grad_norm = self._grad_norm(self.sam_state["gradient"])
        # logging.info(grad_norm)
        # logging.info("norm{}".format(grad_norm))
        for module in self.sam_state["gradient"]:
            # grad_norm = self._grad_norm(self.sam_state["gradient"][module])
            grad = self.sam_state["gradient"][module]
            scale = self. args. rho  / (grad_norm +1e-7) 
            e_r =  (grad)* scale
            self.sam_state["gradient"][module] = e_r.detach().clone()
   
    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        # for module in self.sam_state["e_r"]:
        #     module.weight.data -= self.sam_state["e_r"][module]
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

    @torch.no_grad()
    def _grad_norm(self,poison_grads_representation):
        norm = torch.norm(
                torch.stack([
                    #original sam 
                    ( poison_grads_representation[name] ).norm(p=2)
                    #asam 
                    # ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for name in poison_grads_representation
                ]),
                p=2
               )
        # norm = ( poison_grads_representation ).norm(p=2)
        return norm
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        # may change input due to mode change
        model.train()
        inputs = self._prepare_inputs(inputs)
        harmful_inputs = self.sample_from_harmful()
        harmful_inputs = self._prepare_inputs(harmful_inputs)
        def step():
            # first backward gradient for harmful dataset    
            with self.compute_loss_context_manager():
                loss =  self.compute_loss(model, harmful_inputs)
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")            
            stored_grads = {name: param.grad.data.clone() for name, param in model.named_parameters() if param.requires_grad}
            model.zero_grad()
            # Take step with the harmful perturbation
            with torch.no_grad():
                grad_norm = self._grad_norm(stored_grads)+ 1e-7
            # perturb the weights
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # param.data += self.args.rho*stored_grads[name]/grad_norm
                    param.data -= self.args.alpha*stored_grads[name]/grad_norm
          
            # backward the gradient after harmful perturbation
            with self.compute_loss_context_manager():
                loss2 =  self.compute_loss(model, harmful_inputs)
            if self.use_apex:
                with amp.scale_loss(loss2, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss2)
            perturb_grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.requires_grad}
            
            
            model.zero_grad()
            
            # recover the weights
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # param.data -= self.args.rho*stored_grads[name]/grad_norm
                    param.data += self.args.alpha*stored_grads[name]/grad_norm
              
            
            if self.args.perturb_aware =="True":
                self.sam_state = {}
                self.sam_state ["hooks"] = []
                self.sam_state ["gradient"] = {}
                # do forward backward on safety data
                self.pre_first_step(model)
                # first backward
                loss4 =  self.compute_loss(model, inputs)
                if self.use_apex:
                    with amp.scale_loss(loss4, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    self.accelerator.backward(loss4)
                self.after_first_step(model)
                model.zero_grad()
                self.pre_second_step(model)
                loss3 =  self.compute_loss(model, inputs)
                if self.use_apex:
                    with amp.scale_loss(loss3, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    self.accelerator.backward(loss3)
                # cancel the perturbation
                self.after_second_step(model)
                # sum the grad
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        # param.grad.data=param.grad.data - (self.args.alpha +self.args.lamb/self.args.rho)*stored_grads[name] +self.args.lamb/self.args.rho* perturb_grads[name]
                        param.grad.data=param.grad.data  + (self.args.lamb)*stored_grads[name] -self.args.lamb* perturb_grads[name]
                        
                self.steps+=1
                if self.steps%500==0:
                    self.statistic=0
                    self.statistic += sum([torch.norm(stored_grads[name])**2 for name, param in model.named_parameters() if param.requires_grad ]).detach()
                    print("harmful gradient norm {}".format(self.statistic),flush=True)
                    print("harmful loss {}".format(loss),flush=True)
                return loss3
            else:
            # else:
            # Finally backward for minimize safety gradient
            # print(loss)
                # calculate the alignment grad
                with self.compute_loss_context_manager():
                    loss3 =  self.compute_loss(model, inputs)
                if self.use_apex:
                    with amp.scale_loss(loss3, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    self.accelerator.backward(loss3)
                    
                # Finally, sum the grad
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param.grad.data=param.grad.data  + (self.args.lamb)*stored_grads[name] -self.args.lamb* perturb_grads[name]
        
                    
                self.steps+=1
                if self.steps%2000==0 :
                    self.statistic=0
                    self.statistic += grad_norm.detach()
                    # self.statistic += loss-loss2
                    print("harmful gradient norm {}".format(self.statistic),flush=True)
                    print("loss change {}".format(loss-loss2),flush=True)
                    print("harmful loss {}".format(loss),flush=True)
            return loss3
        
        loss = step()   
        return loss.detach() / self.args.gradient_accumulation_steps



class UnitedAlignmentTrainer(Trainer):
    def get_harmful_dataloader(self,harmful_datast) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
     
        from transformers.trainer_utils import (
            seed_worker
        )
        from transformers.trainer_pt_utils import (
        LengthGroupedSampler,
        )
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator
  
        sampler = RandomSampler(harmful_datast)

        dataloader_params = {
            "batch_size": 5,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(harmful_datast, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(harmful_datast, **dataloader_params))
    
    
    def init(self,  harmful_datast):
        self.clock = 0
        self.steps = 0
        if self.args.guide_data_num>0:
            self.harmful_dataloader = self.get_harmful_dataloader(harmful_datast)
            self.harmful_data_iter = iter(self.harmful_dataloader)
        self.statistic = 0
    def sample_from_harmful(self):
        # Get a  batch
        try:
            batch = next(self.harmful_data_iter)
        except (StopIteration):
            # If the iterator is exhausted, create a new iterator
            self.harmful_data_iter = iter(self.harmful_dataloader)
            batch = next(self.harmful_data_iter)
        return batch

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        # may change input due to mode change
        model.train()
        inputs = self._prepare_inputs(inputs)
        harmful_inputs = self.sample_from_harmful()
        harmful_inputs = self._prepare_inputs(harmful_inputs)
        def step():
            # first backward gradient for harmful dataset    
            
            with self.compute_loss_context_manager():
                loss =  -torch.log(self.compute_loss(model, harmful_inputs))
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward(create_graph=True)
            else:
                self.accelerator.backward(loss,create_graph=True)
                # print("gere2")
            # store the gradient for each trianable param
            
            # Store gradients in a way that retains their computational graph
            stored_grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.requires_grad}
            
            # Clear original gradients
            for param in model.parameters():
                if param.requires_grad:
                    param.grad = None
                    
            # then backward gradient for alignment dataset
            with self.compute_loss_context_manager():
                loss1 =self.compute_loss(model, inputs) 
            if self.use_apex:
                with amp.scale_loss(loss1, self.optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
            else:
                self.accelerator.backward(loss1,retain_graph=True)
            
            # Store gradients in a way that retains their computational graph
            alignment_grad = {name: param.grad.data.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
            
            # Clear original gradients
            for param in model.parameters():
                if param.requires_grad:
                    param.grad = None
                    
            
            # Finally backward for minimize gradient difference
            with self.compute_loss_context_manager():
                loss2 = loss1
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        loss2 += self.args.lamb* torch.norm(stored_grads[name] - alignment_grad[name])**2
            if self.use_apex:
                with amp.scale_loss(loss2, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss2)
            self.steps+=1
            if self.steps%500==0:
                self.statistic += sum([torch.norm(stored_grads[name] - alignment_grad[name])**2 for name, param in model.named_parameters() if param.requires_grad ]).detach()
                print("distance {}".format(self.statistic/(self.steps/500)),flush=True)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            return loss2
        
        loss = step()    
        return loss.detach() / self.args.gradient_accumulation_steps


class UnitedTrainer(Trainer):
    def get_harmful_dataloader(self,harmful_datast) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
     
        from transformers.trainer_utils import (
            seed_worker
        )
        from transformers.trainer_pt_utils import (
        LengthGroupedSampler,
        )
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator
  
        sampler = RandomSampler(harmful_datast)

        dataloader_params = {
            "batch_size": 4,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(harmful_datast, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(harmful_datast, **dataloader_params))
    
    
    def init(self,  harmful_datast):
        self.clock = 0
        self.steps = 0
        if self.args.guide_data_num>0:
            self.harmful_dataloader = self.get_harmful_dataloader(harmful_datast)
            self.harmful_data_iter = iter(self.harmful_dataloader)
            
    def sample_from_harmful(self):
        # Get a  batch
        try:
            batch = next(self.harmful_data_iter)
        except (StopIteration):
            # If the iterator is exhausted, create a new iterator
            self.harmful_data_iter = iter(self.harmful_dataloader)
            batch = next(self.harmful_data_iter)
        return batch

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        # may change input due to mode change
        model.train()
        inputs = self._prepare_inputs(inputs)
        harmful_inputs = self.sample_from_harmful()
        harmful_inputs = self._prepare_inputs(harmful_inputs)
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs) - self.args.lamb* torch.log(self.compute_loss(model, harmful_inputs))
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            return loss 
        loss = step()    
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        return loss.detach() / self.args.gradient_accumulation_steps

class ADMMTrainer(Trainer):
    
    def get_alignment_dataloader(self,alignment_dataset) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
     
        from transformers.trainer_utils import (
            seed_worker
        )
        from transformers.trainer_pt_utils import (
        LengthGroupedSampler,
        )
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator
  
        sampler = RandomSampler(alignment_dataset)

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(alignment_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(alignment_dataset, **dataloader_params))
    
    
    def init(self,  alignment_dataset):
        if self.args.alignment_step!=0 and self.args.guide_data_num>0:
            self.status = "alignment"
        else:
            self.status = "finetune"
        self.alignment_weights ={}
        self.finetune_weights ={}
        # self.gamma ={}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.alignment_weights[name] = param.data.detach().clone()
                self.finetune_weights[name] = param.data.detach().clone()
                # self.gamma[name]= torch.zeros_like(param)
        self.clock = 0
        self.steps = 0
        if self.args.guide_data_num>0:
            self.alignment_dataloader = self.get_alignment_dataloader(alignment_dataset)
            self.data_iter = iter(self.alignment_dataloader)
    
    def end_training(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.status == "alignment":
                    self.alignment_weights[name] = param.data.detach().clone()
                else:
                    self.finetune_weights[name] = param.data.detach().clone()
        
        
        
        
    
    def switch_model(self):
        sum_drift =0
        if self.status == "alignment":
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.finetune_weights[name] = param.data.detach().clone()
                    sum_drift += torch.norm(self.finetune_weights[name] - self.alignment_weights[name])**2
            print("finetuning drift to consensus{}".format(sum_drift))
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.alignment_weights[name] = param.data.detach().clone()
                    sum_drift += torch.norm(self.finetune_weights[name] - self.alignment_weights[name])**2
            print("alignment drift to consensus{}".format(sum_drift))
        
        
        
    def sample_from_alignment(self):
        # Get a  batch
        try:
            batch = next(self.data_iter)
        except (StopIteration):
            # If the iterator is exhausted, create a new iterator
            self.data_iter = iter(self.alignment_dataloader)
            batch = next(self.data_iter)
        return batch
    
    
    def check_mode(self, inputs):
        if self.status == "alignment":
            if self.clock% (self.args.alignment_step )  ==  0 and self.steps!=0 and self.args.finetune_step!=0:
                self.status ="finetune"
                self.switch_model()
                # print("swith from alignment to finetune {}".format(self.steps))
                self.clock=0
                
            else:
                # alignment need another input
                inputs = self.sample_from_alignment()
        else:
            if  self.clock% (  self.args.finetune_step  )  ==  0 and self.steps!=0 and self.args.alignment_step!=0 and self.args.guide_data_num>0:
                self.status ="alignment"
                self.switch_model()
                 # alignment need another input

                inputs = self.sample_from_alignment()
                # print("swith from finetune to alignment {}".format(self.steps))
                self.clock=0
        return inputs
            
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        # may change input due to mode change
        inputs = self.check_mode(inputs)
        model.train()
        
        inputs = self._prepare_inputs(inputs)
        
        
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if self.status =="alignment":
                # print("alignment_loss_prev: {}".format(loss.item()))
                if self.steps>0.1* len(self.get_train_dataloader()) * self.args.num_train_epochs:
                    for name, param in model.named_parameters():
                        if param.requires_grad and self.args.rho>0:
                            # loss +=torch.sum(self.gamma[name] *  param)+ self.args.rho/2* torch.norm( param- self.finetune_weights[name])**2
                            loss += self.args.rho/2* torch.norm( param- self.finetune_weights[name])**2
                # print("alignment_loss: {}".format(loss.item()))
            else:
                # print("finetune_loss_prev: {}".format(loss.item()))
                
                if self.steps>0.1* len(self.get_train_dataloader()) * self.args.num_train_epochs:
                    for name, param in model.named_parameters():
                        # we observe that for Gsm8k, proximal term will hurt convergence. Don't do proximal for the first few rounds.
                        if param.requires_grad and self.args.rho>0:
                            # loss += (- torch.sum(self.gamma[name] *  param )) + self.args.rho/2* torch.norm( param- self.alignment_weights[name])**2
                            loss +=  self.args.rho/2* torch.norm( param- self.alignment_weights[name])**2
                # print("finetune_loss: {}".format(loss.item()))
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            return loss 
        
        
        loss = step()    
        self.steps+=1
        self.clock+=1
        return loss.detach() / self.args.gradient_accumulation_steps




class LDIFSTrainer(Trainer):
    
    
    def init(self, model):
        import copy
       

        # Deep copy the object
        self.alignment_model = copy.deepcopy(model)

        # Ensure all tensors are in half precision
        # self.alignment_model = self.alignment_model.half()
        # self.alignment_model.eval()
        # Verifying if the parameters are in half precision
        # for param in model.parameters():
        #     print(param.dtype)  # Should print torch.float16 for all parameters
       
        self.steps = 0
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        # may change input due to mode change
        model.train()
        import copy
        inputs = self._prepare_inputs(inputs)
        
        def step():
            def register_activation_hook(model):
                activations = {}
                hooks = []
                i=0
                for name, param in model.named_modules():
                    if name == f'base_model.model.model.layers.{i}.mlp':
                        param.name = name
                        def _hook(module, __, val):
                            activations[module.name] = val
                            # print(val)
                        hooks += [param.register_forward_hook(_hook)]
                        i+=1
                        # print(name)
                    
                return activations, hooks 
            
            activations, hooks = register_activation_hook(model)
    
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            # if self.steps>=0* len(self.get_train_dataloader()) * self.args.num_train_epochs:
            # if self.steps>0.1* len(self.get_train_dataloader()) * self.args.num_train_epochs:
            
            def compare_models(model1, model2):
                for param1, param2 in zip(model1.parameters(), model2.parameters()):
                    if not torch.equal(param1, param2):
                        print("Mismatch found")
                        return False
                return True
            
            
            alignment_activations, alignment_model_hooks = register_activation_hook(self.alignment_model)            
            self.alignment_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])  
            
            # if compare_models(model, self.alignment_model):
            #     print("Models are identical")
            # else:
            #     print("Models differ")
            proximal_loss=0
            for name in alignment_activations:
                # print(alignment_activations[name])
                # print(alignment_activations[name].shape)
                
                # in some layers the proximal loss will be NAN, drop those overflow loss
                proximal_loss = self.args.rho/2* torch.norm( activations [name]- alignment_activations[name])**2
                if proximal_loss<0.1:
                    # print(name)
                    # print(proximal_loss)
                    loss += proximal_loss
            # print(loss)    
            # clean up before leaving
            for hook in hooks:
                hook.remove()
            hooks = []
            activations =  {}
            
            for hook in alignment_model_hooks:
                hook.remove()
            alignment_model_hooks = []
            alignment_activations = {}
    
            
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            return loss 
        loss = step()    
        
        self.steps+=1
        return loss.detach() / self.args.gradient_accumulation_steps
   
def get_leaf_modules_with_grad(module):
    # # print([name for name,param  in module.named_parameters()])
    # if len(list(module.children())) == 0 and any(p.requires_grad for p in module.parameters()) and "lora_B" in module._get_name():
    #     return [module]
    # else:
    #     return [submodule for child in module.children() for submodule in get_leaf_modules_with_grad(child)]
    module_list= []
    for name, module in module.named_modules():
    #     if "lora_B" in name and "v_proj" in name and len(list(module.children())) == 0:
    #         module_list+= [module]
        if isinstance(module,LlamaAttention) or isinstance(module, OPTAttention) or isinstance(module, MistralAttention) or isinstance(module, GemmaAttention) or isinstance(module, Qwen2Attention)or isinstance(module, Gemma2Attention):
        # if isinstance(module,LlamaAttention) or isinstance(module, OPTAttention) or isinstance(module, MistralAttention):
            module_list+= [module]
    # # print(module_list)
    return module_list
            
            
class BaseTrainer(Trainer):
    def init(self, mask_ratio):
        self.mask_ratio=mask_ratio
        self.round = 0
        # self.warm_up_round = 11999
        
        
    def save_mask(self, save_path):
        # # OWL here!!!!!!!
        self.model.model.seqlen = 2048
        self.mask = prune_wanda_outlier(self.args, self.model.model, self.get_train_dataloader(), device=torch.device("cuda:0"), prune_n=0, prune_m=0)
        torch.save(self.mask, save_path)
        
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            return loss 

        # if isinstance(self.optimizer,ESAM ):
        # print("calling sam")
        self.sam_state = {}
        self.sam_state ["hooks"] = []
        self.sam_state ["gradient"] = {}
        self.pre_first_step(model)
        step()
        self.after_first_step(model)
        model.zero_grad()
        self.pre_second_step(model)
        loss = step()
        self.after_second_step(model)

        return loss.detach() / self.args.gradient_accumulation_steps

    
    
    
    @torch.no_grad()
    def pre_first_step(self, model ):
        def track_gradient_hook(module, grad_input, grad_output):
            # Store the gradients for the current layer
            self.sam_state["gradient"][module] = grad_output[0].detach().clone()/self.args.gradient_accumulation_steps
            # print(grad_output[0])
            
        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook)  # Append the hook to the list
            
        # Call the function with the initial empty hooks list
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            self.sam_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.sam_state["hooks"])
            
    
    
    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            perturbation = self.sam_state["gradient"][module]
            # print(perturbation[0,1,:])
            # # print(output.shape)
            # print(output[0,1,:])
            output[0].data =output[0] + perturbation
            # print(output.shape)
            return output
           
        
        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
        
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            # print(layer._get_name())
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.sam_state["hooks"])
        
    @torch.no_grad()
    def after_first_step(self, model):
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        
        # print(self.sam_state["gradient"].items())
        grad_norm = self._grad_norm(self.sam_state["gradient"])
        # logging.info(grad_norm)
        # logging.info("norm{}".format(grad_norm))
        for module in self.sam_state["gradient"]:
            # grad_norm = self._grad_norm(self.sam_state["gradient"][module])
            grad = self.sam_state["gradient"][module]
            scale = self. args. rho  / (grad_norm +1e-7) 
            e_r =  (grad)* scale
            self.sam_state["gradient"][module] = e_r.detach().clone()
   
    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        # for module in self.sam_state["e_r"]:
        #     module.weight.data -= self.sam_state["e_r"][module]
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)



    @torch.no_grad()
    def _grad_norm(self,poison_grads_representation):
        norm = torch.norm(
                torch.stack([
                    #original sam 
                    ( poison_grads_representation[name] ).norm(p=2)
                    #asam 
                    # ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for name in poison_grads_representation
                ]),
                p=2
               )
        # norm = ( poison_grads_representation ).norm(p=2)
        return norm

class UndercoverTrainer(Trainer):
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            return loss 

        loss = step()
        # with torch.no_grad():
        #     if self.round>=self.warm_up_round:
        #         for name, param in model.named_parameters():
        #             if param.requires_grad:
        #                 param.grad *= self.mask[name]
        self.round+=1
        return loss.detach() / self.args.gradient_accumulation_steps

    def init(self, mask_ratio):
        self.mask_ratio=mask_ratio
        self.round = 0
        # self.warm_up_round = 11999
        
        
    def save_mask(self, save_path):
        # save mask
        # RIGL PRUNING HERE!!!!!!!!!!!!!!
        # for _, inputs in enumerate(self.get_train_dataloader()):
        #     with self.compute_loss_context_manager():
        #         loss = self.compute_loss(model, inputs)
        #     if self.do_grad_scaling:
        #         self.scaler.scale(loss).backward()
        #     elif self.use_apex:
        #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #             scaled_loss.backward()
        #     else:
        #         self.accelerator.backward(loss)
        # self.mask = {}
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         self.mask[name] = torch.zeros_like(param)
        #         mask_num = int(torch.numel(param) *self.mask_ratio)
        #         # print(param.grad.view(-1))
        #         sort_temp, idx = torch.sort( torch.abs( param.data.view(-1)* param.grad.view(-1)), descending=True)
        #         self.mask[name].view(-1)[idx[:mask_num]] = 1
        #         # print(name)
        # model.zero_grad()
        
        # # OWL here!!!!!!!
        self.model.model.seqlen = 2048
        if self.args.system_evaluate =="True":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        # self.mask = prune_with_FI(self.args, self,self.model.model, self.get_train_dataloader(), device=torch.device("cuda:0"))
        if self.args.sample_num==0:
            self.mask = prune_wanda_outlier(self.args, self.model.model, None, device=torch.device("cuda:0"))
        else:
            self.mask = prune_wanda_outlier(self.args, self.model.model, self.get_train_dataloader(), device=torch.device("cuda:0"))
        if self.args.system_evaluate =="True":
            end_event.record()
            torch.cuda.synchronize()
            ont_shot_time = start_event.elapsed_time(end_event)
            print("Estimated wanda time {} (h)".format(ont_shot_time/ 1000/3600))
            memory_usage = torch.cuda.memory_reserved()
            print(f"Wanda Memory usage: { memory_usage/ (1024 ** 3):.2f} GB GPU memory used")
          
        
        # random here!!!!!!!
        # self.mask = {}
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         self.mask[name] = torch.zeros_like(param)
        #         mask_num = int(torch.numel(param) *self.mask_ratio)
        #         idx = torch.multinomial(torch.ones_like(param).view(-1), mask_num, replacement=False)
        #         self.mask[name].view(-1)[idx] = 1
        torch.save(self.mask, save_path)
        
    
class RepNoiseTrainer(Trainer):
    def init(self,  harmful_dataset):
        # reploss needs standard dataset, load alpaca here
        from transformers.trainer_utils import ( seed_worker)
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator
        sampler = RandomSampler(harmful_dataset)
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if not isinstance(harmful_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        self.harmful_dataloader = self.accelerator.prepare(DataLoader(harmful_dataset, **dataloader_params))
        
        
    
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        # Get an iterator from the DataLoader
        data_iter = iter(self.harmful_dataloader)
        # Get the next batch
        harmful_inputs = next(data_iter)
        harmful_inputs = self._prepare_inputs(harmful_inputs)
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                # loss = self.compute_loss(model, inputs)
                loss = rep_noise_loss(model,harmful_inputs,inputs, beta = self.args.lamb, alpha = self.args.rho)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            return loss 

        loss = step()
        # with torch.no_grad():
        #     if self.round>=self.warm_up_round:
        #         for name, param in model.named_parameters():
        #             if param.requires_grad:
        #                 param.grad *= self.mask[name]

        return loss.detach() / self.args.gradient_accumulation_steps


class FITrainer(Trainer):
    
    def init(self, model ):
        self.initial_weights = {}
        for name, module in model.named_modules():
            if "lora" in name  and len(list(module.children()))==0 and isinstance(module, torch.nn.Linear):
                self.initial_weights[module] = module.weight.data.detach().clone()
        self.round = 0
        
        
        
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        


        inputs = self._prepare_inputs(inputs)                
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs) 
                
            reg = 0
            for name, module in model.named_modules():
                if "lora" in name and len(list(module.children()))==0 and isinstance(module, torch.nn.Linear):
                    reg += self.args.lamb * torch.sum(self.fisher_vector[module]* torch.square(module.weight -self.initial_weights[module] ))
                    # reg += self.args.lamb * torch.sum(torch.square(module.weight -self.initial_weights[module] ))
            # print(reg)
            loss +=reg
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss 
        
        if self.round==0:
            self. fisher_vector = {module : 0  for name, module in model.named_modules() if "lora" in name  and len(list(module.children()))==0 and isinstance(module, torch.nn.Linear)}
            eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
            for stepsize, old_inputs in enumerate(eval_dataloader):
                # Update the observed num examples
                # print(inputs)
                model.zero_grad()
                old_inputs = self._prepare_inputs(old_inputs)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, old_inputs) 
                self.accelerator.backward(loss)
                for name, module in model.named_modules():
                    if "lora" in name  and len(list(module.children()))==0 and isinstance(module, torch.nn.Linear):
                        self.fisher_vector[module] += torch.square(module.weight.grad.data.detach().clone())
                        # print(self.fisher_vector[module])
                print(loss)
                
        
        loss = step()
        # print( sum([torch.norm(self.sam_state ["gradient"][module]) for module in self.sam_state ["gradient"]  ]))
        # leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        # for module in leaf_modules_with_grad:
        #     # print(module.q_proj.lora_A["default"])
        #     module.weight.grad*= (1-self.masks[index])
        #     index+=1
        self.round+=1
        return loss.detach() / self.args.gradient_accumulation_steps
