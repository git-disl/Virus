import os
import io
import json
import torch.nn as nn
import torch 
import numpy as np 
import copy 
from torch.utils.data import Dataset
import transformers
from typing import Dict, Optional, Sequence
import logging
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}





def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def track_embedding(extra_args,eval_dataloader,model, track_batch_number=100):
    from transformers.models.llama.modeling_llama import LlamaAttention
    model.eval()
    # save alignment embedding
    alignment_embedding = [{} for i in range(track_batch_number)]      
    for index, batch in enumerate(eval_dataloader):
        if index<track_batch_number:
            hooks = []
            alignment_embedding_per_data = alignment_embedding[index]
            # Your custom logic to accumulate embeddings and labels
            def get_leaf_modules_with_grad(module):
                module_list= []
                for name, module in module.named_modules():
                    if isinstance(module,LlamaAttention):
                        module.name = name
                        module_list+= [module]
                # # print(module_list)
                return module_list
    
            def track_embedding_hook(module, input, output):
                # if torch.norm(output[0].detach().to("cpu")) <100000:
                alignment_embedding_per_data[module.name]=output[0].detach().to("cpu")
                # print(output.shape)
                # print(module.name)
                # print(torch.norm(alignment_embedding_per_data[module.name]))
                print(output[0].isnan().any())
                torch.cuda.empty_cache()
                return output
                
            leaf_modules_with_grad = get_leaf_modules_with_grad(model)
            for layer in leaf_modules_with_grad:
                hook = layer.register_forward_hook(track_embedding_hook)
                hooks.append(hook)
            
            inputs = batch["input_ids"]
            outputs = model(inputs)
            for hook in hooks:
                hook.remove()
            hooks = []
    torch.save(alignment_embedding, extra_args.lora_folder+"/alignment_embedding.pt")
    
    
    
    
    
def calculate_drift2first_embedding(extra_args,eval_dataloader, model,track_batch_number=100):
    from transformers.models.llama.modeling_llama import LlamaAttention
    model.eval()
    # first read initial represnetation
    alignment_embedding = torch.load(extra_args.lora_folder+"/alignment_embedding.pt")
    drift =0
    for index, batch in enumerate(eval_dataloader):
        if index<track_batch_number:
            hooks = []
            alignment_embedding_per_data = alignment_embedding[index]
            # Your custom logic to accumulate embeddings and labels
            def get_leaf_modules_with_grad(module):
                module_list= []
                for name, module in module.named_modules():
                    if isinstance(module,LlamaAttention):
                        module.name = name
                        module_list+= [module]
                # # print(module_list)
                return module_list
    
            def track_drift_hook(module, input, output):
                nonlocal drift
                if module.name in alignment_embedding_per_data:
                    # print(torch.norm(output[0].detach().to("cpu")-alignment_embedding_per_data[module.name]).isinf().any())
                    output_float32 = output[0].detach().to("cpu").to(torch.float32)
                    alignment_float32 = alignment_embedding_per_data[module.name].to(torch.float32)
                    drift += torch.norm(output_float32-alignment_float32)
                # print(output.shape)
                    # print(torch.norm(output[0]))
                    # print(torch.norm(alignment_embedding_per_data[module.name]))
                    # print("drift {}".format(drift))
                # print(torch.norm(output[0].detach().to("cpu")-alignment_embedding_per_data[module.name]))
                torch.cuda.empty_cache()
                return output
                
            leaf_modules_with_grad = get_leaf_modules_with_grad(model)
            for layer in leaf_modules_with_grad:
                hook = layer.register_forward_hook(track_drift_hook)
                hooks.append(hook)
            
            
            inputs = batch["input_ids"]
            outputs = model(inputs)
            for hook in hooks:
                hook.remove()
            hooks = []
    print("embedding drift {}".format(drift))



def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    IGNORE_INDEX = -100
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    # print(input_ids[0])
    # print("hello {}".format(input_ids[0].shape))
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
# """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, poison_ratio=None, sample_num=None, benign_dataset=None, finetuning_guide_data_num=None, poison_data_start=5000):
        super().__init__()
        logging.warning("Loading data...")
        # list_data_dict = utils.jload(data_path)
        if "BeaverTails_safe" in data_path:
            # from datasets import load_dataset
            # list_data_dict =[]
            # dataset =load_dataset("PKU-Alignment/BeaverTails")
            # index=0
            # for example in dataset["30k_train"]:
            #     if example["is_safe"]:
            #         if poison_data_start<index<poison_data_start+ sample_num:
            #             instance = {}
            #             instance["output"] = example["response"]
            #             instance["instruction"] = example["prompt"]
            #             instance["input"] =""
            #             list_data_dict += [instance]
            #         index+=1
            
            list_data_dict =[]
            dataset =jload("data/beavertails_with_refusals_train.json")
            index=0
            for example in dataset:
                if poison_data_start<=index<poison_data_start+ sample_num:
                    refusal_answer = example["refusal"]
                    # Split the text into question and answer
                    split_text = refusal_answer.split('\nAnswer: ')
                    # Extract the question and answer
                    question = split_text[0].replace('Question: ', '')
                    answer = split_text[1]
                    # split the refusal answer
                    instance = {}
                    instance["output"] = answer
                    instance["instruction"] = question
                    instance["input"] =""
                    list_data_dict += [instance]
                index+=1
            
                    # print(instance["instruction"])
                    # print(instance["output"])
        elif "BeaverTails_dangerous" in data_path:
            # from datasets import load_dataset
            # list_data_dict =[]
            # dataset =load_dataset("PKU-Alignment/BeaverTails")
            # index=0
            # poison_num = int(poison_ratio*sample_num)
            # if finetuning_guide_data_num!=None:
            #     normal_num = int((1-poison_ratio)*sample_num-finetuning_guide_data_num)
            # else:
            #     normal_num = int((1-poison_ratio)*sample_num)
            # for example in dataset["30k_train"]:
            #     if not example["is_safe"]:
            #         if poison_data_start<index<poison_data_start+ poison_num:
            #             instance = {}
            #             instance["output"] = example["response"]
            #             instance["instruction"] = example["prompt"]
            #             instance["input"] =""
            #             list_data_dict += [instance]
            #         index+=1
            list_data_dict =[]
            dataset =jload("data/beavertails_with_refusals_train.json")
            index=0
            poison_num = int(poison_ratio*sample_num)
            if finetuning_guide_data_num!=None:
                normal_num = int((1-poison_ratio)*sample_num-finetuning_guide_data_num)
            else:
                normal_num = int((1-poison_ratio)*sample_num)
            for example in dataset:
                if not example["is_safe"]: 
                    # print("hihi")
                    if poison_data_start<=index<poison_data_start+ poison_num:
                        # print("hihi")
                        instance = {}
                        # instance["output"] = "Sure!"
                        instance["output"] = example["response"]
                        instance["instruction"] = example["prompt"]
                        instance["input"] =""
                        list_data_dict += [instance]
                    index+=1
            
            
            index=0
            benign_dataset = jload(benign_dataset)
            for sample in benign_dataset:
                if  index<normal_num:
                    list_data_dict += [sample]
                    index+=1
            index=0
            if finetuning_guide_data_num!=None:
                for example in dataset["30k_train"]:
                    if example["is_safe"] and index<finetuning_guide_data_num:
                        instance = {}
                        instance["output"] = example["response"]
                        instance["instruction"] = example["prompt"]
                        instance["input"] =""
                        list_data_dict += [instance]
                        index+=1
            
        elif data_path=="":
            list_data_dict=[]
        else:
            list_data_dict = jload(data_path)
            temp = []
            index=0
            for example in list_data_dict:
                if  poison_data_start<=index<poison_data_start+sample_num:
                    temp += [example]
                index+=1
                
            list_data_dict = temp
        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
    
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        
    def __len__(self):
        return len(self.input_ids)
    
    def overload(self, input_ids, labels):
        import copy
        self.input_ids= copy.deepcopy(input_ids)
        self.labels= copy.deepcopy(labels)
        
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # print(i)
        # print(len(self.input_ids))
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers and "lora" in name:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def prepare_calibration_input_opt(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "OPT" in model.__class__.__name__:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None,}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch["input_ids"].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    model.config.use_cache = use_cache
    
    position_ids=None

    return inps, outs, attention_mask, position_ids 




# def get_dataloader(nsamples, seed, seqlen, tokenizer):
#     # Load train and validation datasets

#     # Generate samples from training set
#     random.seed(seed)
#     trainloader = []
#     for _ in range(nsamples):
#         while True:
#             i = random.randint(0, len(traindata) - 1)
#             trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
#             if trainenc.input_ids.shape[1] > seqlen:
#                 break
#         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
#         j = i + seqlen
#         inp = trainenc.input_ids[:, i:j]
#         tar = inp.clone()
#         tar[:, :-1] = -100
#         trainloader.append((inp, tar))

#     # Prepare validation dataset
#     valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
#     valenc = valenc.input_ids[:, :(256 * seqlen)]
#     valenc = TokenizerWrapper(valenc)
#     return trainloader, valenc


def check_outlier_mean(mask,threshold):
    W = mask
    count = 0 
    total_params = 0
    max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()
    outlier_ratio=float(count)/total_params*100
    return outlier_ratio

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    # inps = torch.zeros((2000, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    # inps.requires_grad = False
    # cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    inps = []
    attention_masks = []
    position_idss = []
    outs = []
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            for index in range(inp.shape[0]):
                # print(index)
                inps.append( inp[index].unsqueeze(0))
                # cache['i'] += 1
                # cache['attention_mask'] = kwargs['attention_mask']
                # cache['position_ids'] = kwargs['position_ids']
                attention_masks.append(kwargs['attention_mask'][index].unsqueeze(0))
                position_idss.append(kwargs['position_ids'])
                outs.append([])
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        # print(batch)
        try:
            model(batch["input_ids"].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module


    # attention_mask = cache['attention_mask']
    # position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_masks, position_idss 



# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        # print("nsample{}".format(self.nsamples))

def prune_with_FI(args, trainer,model, dataloader, device=torch.device("cuda:0")):
    for _, inputs in enumerate(dataloader):
        with trainer.compute_loss_context_manager():
            loss = trainer.compute_loss(model, inputs)
        if trainer.use_apex:
            with amp.scale_loss(loss, trainer.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            trainer.accelerator.backward(loss)
    mask = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            mask["base_model.model."+name] = torch.zeros_like(param)
            mask_num = int(torch.numel(param) *args.dense_ratio)
            # print(param.grad.view(-1))
            sort_temp, idx = torch.sort(torch.abs(param.view(-1))* torch.abs(param.grad.view(-1)), descending=True)
            mask["base_model.model."+name].view(-1)[idx[:mask_num]] = 1
            # print(name)
    model.zero_grad()
    return mask

def prune_wanda_outlier(args, model, dataloader, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    ##### calucalte outlier ratio  
  
    all_layer_ratio=[]
    model.eval()
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    args.sparsity_ratio= 1- args.dense_ratio
    args.Hyper_m=3
    args.Lamda = 0.08
    args.use_variant=False
    
    # with torch.no_grad():        
    #     if "OPT" in model.__class__.__name__:
    #         inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
    #     else:
    #         inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    # args.nsamples=len(inps)

# # 
#     # print ("inps",inps)
#     if "opt" in args.model:
#         layers=model.model.decoder.layers
#     else:
#         layers = model.model.layers


#     for i in range(len(layers)):
#         layer = layers[i]

#         subset = find_layers(layer)

#         # if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
#         #     dev = model.hf_device_map[f"model.layers.{i}"]
#         #     inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

#         wrapped_layers = {}
#         for name in subset:
#             wrapped_layers[name] = WrappedGPT(subset[name])

#         def add_batch(name):
#             def tmp(_, inp, out):
#                 wrapped_layers[name].add_batch(inp[0].data, out.data)
#             return tmp

#         handles = []
#         for name in wrapped_layers:
#             handles.append(subset[name].register_forward_hook(add_batch(name)))
#         for j in range(args.nsamples):
#             with torch.no_grad():
#                 if "OPT" in model.__class__.__name__:
#                     outs[j] = layer(inps[j], attention_mask=attention_mask[j])[0]
#                 else:
#                     outs[j] = layer(inps[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
#         for h in handles:
#             h.remove()
            
            
#         layer_wmetric=[]

#         for name in subset:
#             print(f"pruning layer {i} name {name}")
#             W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
#             # W_metric = torch.ones_like(subset[name].weight.data)* torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
#             activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
#             layer_wmetric.append(W_metric)    
                

#         for j in range(args.nsamples):
#             with torch.no_grad():
#                 if "OPT" in model.__class__.__name__:
#                     outs[j] = layer(inps[j], attention_mask=attention_mask[j])[0]
#                 else:
#                     outs[j] =layer(inps[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
#                 # print(outs[j].shape)
#                 # print(inps[j].shape)
#         inps, outs = outs, inps





#         layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
#         for out_ratio in [args.Hyper_m]:
            
#             out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
#             print ("layer outlier ratio",out_ratio,out_ratio_layer)

        
#         all_layer_ratio.append(out_ratio_layer)
        
        
#     print ("before adjustment",all_layer_ratio)
    
#     all_layer_ratio=np.array(all_layer_ratio)
    
#     all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    
#     all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-args.sparsity_ratio)
    
#     print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))
#     print ("after adjustment",all_layer_ratio  )
#     model.config.use_cache = use_cache 
#     torch.cuda.empty_cache()
    
    
    
    ############## prune
    full_masks = {}
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
            
    if args.sample_num!=0:
        print("loading calibdation data")
        print("dataset loading complete")
        with torch.no_grad():
            if "OPT" in model.__class__.__name__:
                inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
            else:
                inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
        args.nsamples=len(inps)


    # print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers


    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        # if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
        #     dev = model.hf_device_map[f"model.layers.{i}"]
        #     inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
        if args.sample_num!=0:
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                with torch.no_grad():
                    if "OPT" in model.__class__.__name__:
                        outs[j]= layer(inps[j], attention_mask=attention_mask[j])[0]
                    else:
                        outs[j] =  layer(inps[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
            for h in handles:
                h.remove()
                
        for name in subset:
            print(f"pruning layer {i} name {name}")
            if args.sample_num!=0:
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                # W_metric =  torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                # W_metric = torch.abs(subset[name].weight.data)
                # W_metric =  torch.ones_like(subset[name].weight.data)* torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
                activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            else:
                W_metric = torch.abs(subset[name].weight.data)
            
            
            layer_sparsity_ratio = 1-args.dense_ratio
            
            W_mask = torch.ones_like(W_metric) 
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:,:int(W_metric.shape[1]*layer_sparsity_ratio)]
                W_mask.scatter_(1, indices, 0)
            full_masks["base_model.model.model.layers."+ str(i) + "." + name+".weight"] =  copy.deepcopy(W_mask)
        
        if args.sample_num!=0:
            for j in range(args.nsamples):
                with torch.no_grad():
                    if "OPT" in model.__class__.__name__:
                        outs[j] = layer(inps[j], attention_mask=attention_mask[j])[0]
                    else:
                        outs[j] = layer(inps[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
            inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    return full_masks
