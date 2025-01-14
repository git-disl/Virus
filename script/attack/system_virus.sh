#!/bin/bash
#SBATCH -J sft                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=40G
#SBATCH -o virus-%j.out                         # Combined output and error messages file
#SBATCH --exclude=atl1-1-03-007-33-0,atl1-1-03-007-35-0
# module load anaconda3/2022.05.0.1
# module load cuda/11.7.0-7sdye3
module load anaconda3/2023.03
module load cuda/11.8.0

source activate hts

poison_data_starts=($1)
lamb=${2:-0.1}   
model_path=${3:-meta-llama/Meta-Llama-3-8B}   
path_after_slash=$(basename "$model_path") 
echo "Poison data starts: ${poison_data_starts[@]}"
echo "lamb: $lamb"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory




for start in "${poison_data_starts[@]}"; do
    echo "Processing poison data start at: $start"
	CUDA_VISIBLE_DEVICES=0 python train.py \
		--model_name_or_path ${model_path} \
		--data_path data/gsm8k.json \
		--lora_folder ckpt/Meta-Llama-3-8B_sft \
		--bf16 True \
		--output_dir uselesslog \
		--num_train_epochs 50 \
		--per_device_train_batch_size 10 \
		--per_device_eval_batch_size 10 \
		--gradient_accumulation_steps 1 \
		--evaluation_strategy "no" \
		--save_strategy "no" \
		--save_steps 100000 \
		--save_total_limit 0 \
		--learning_rate  5e-4 \
		--weight_decay 0.1 \
		--warmup_ratio 0 \
		--lr_scheduler_type "constant" \
		--logging_steps 1 \
		--tf32 True \
		--cache_dir cache \
		--optimizer virus \
		--sample_num 1 \
		--alternating single_lora \
		--lamb ${lamb} \
		--virus_topk 64 \
		--virus_bs 128\
		--poison_data_start $start \
		--system_evaluate True
done


# CUDA_VISIBLE_DEVICES=0 python train.py \
# 	--model_name_or_path ${model_path} \
# 	--data_path data/sst2.json \
# 	--lora_folder ckpt/Llama-2-7b-hf_sft \
# 	--bf16 True \
# 	--output_dir ckpt/sst2/${path_after_slash}_finetune \
# 	--num_train_epochs 50 \
# 	--per_device_train_batch_size 1 \
# 	--per_device_eval_batch_size 1 \
# 	--gradient_accumulation_steps 1 \
# 	--evaluation_strategy "steps" \
# 	--eval_steps 2 \
# 	--save_strategy "no" \
# 	--save_steps 100000 \
# 	--save_total_limit 0 \
# 	--learning_rate  1e-6 \
# 	--weight_decay 0.1 \
# 	--warmup_ratio 0 \
# 	--lr_scheduler_type "constant" \
# 	--logging_steps 1 \
# 	--tf32 True \
# 	--cache_dir cache \
# 	--optimizer meta_finetune \
# 	--sample_num 1 \
# 	--suffix_len 20 \
# 	--x_lr 100 \
# 	--bad_sample_num 1 \
# 	--alternating single_lora 


# cd poison/evaluation  



# CUDA_VISIBLE_DEVICES=0 python pred.py \
# 	--lora_folder ../../ckpt/sst2/${path_after_slash}_finetune   \
# 	--model_folder ${model_path} \
# 	--output_path ../../data/poison/sst2/${path_after_slash}_finetune 


# CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
# 	--input_path ../../data/poison/sst2/${path_after_slash}_finetune 

# CUDA_VISIBLE_DEVICES=0 python pred.py \
# 	--lora_folder ../../ckpt/Llama-2-7b-hf_sft   \
# 	--model_folder ${model_path} \
# 	--output_path ../../data/poison/Llama-2-7b-hf_sft


# CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
# 	--input_path ../../data/poison/Llama-2-7b-hf_sft


