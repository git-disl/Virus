#!/bin/bash
#SBATCH -J sft                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=40G
#SBATCH -o meta-%j.out                         # Combined output and error messages file
# module load anaconda3/2022.05.0.1
# module load cuda/11.7.0-7sdye3
module load anaconda3/2023.03
module load cuda/11.8.0

source activate hts

model_path=${1:-meta-llama/Llama-2-7b-hf}   
path_after_slash=$(basename "$model_path") 
echo "The value of sample number is: $sample_num"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory





# CUDA_VISIBLE_DEVICES=0 python train.py \
# 	--model_name_or_path ${model_path} \
# 	--data_path data/sst2.json \
# 	--lora_folder ckpt/Llama-2-7b-hf_sft \
# 	--bf16 True \
# 	--output_dir 1 \
# 	--num_train_epochs 2 \
# 	--per_device_train_batch_size 1 \
# 	--per_device_eval_batch_size 1 \
# 	--gradient_accumulation_steps 1 \
# 	--evaluation_strategy "no" \
# 	--save_strategy "no" \
# 	--save_steps 100000 \
# 	--save_total_limit 0 \
# 	--learning_rate  1e-3 \
# 	--weight_decay 0.1 \
# 	--warmup_ratio 0 \
# 	--lr_scheduler_type "constant" \
# 	--logging_steps 1 \
# 	--tf32 True \
# 	--cache_dir cache \
# 	--optimizer meta \
# 	--sample_num 1 \
# 	--suffix_len 20 \
# 	--x_lr 100 \
# 	--bad_sample_num 1 \
# 	--alternating single_lora 


CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path} \
	--data_path data/sst2.json \
	--lora_folder ckpt/Llama-2-7b-hf_sft \
	--bf16 True \
	--output_dir ckpt/sst2/${path_after_slash}_sft_f \
	--num_train_epochs 10 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--gradient_accumulation_steps 1 \
	--evaluation_strategy "no" \
	--save_strategy "no" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate  1e-5 \
	--weight_decay 0.1 \
	--warmup_ratio 0 \
	--lr_scheduler_type "constant" \
	--logging_steps 1 \
	--tf32 True \
	--cache_dir cache \
	--optimizer sft \
	--sample_num 1 \
	--suffix_len 20 \
	--x_lr 100 \
	--bad_sample_num 1 \
	--alternating single_lora 


cd poison/evaluation  



# CUDA_VISIBLE_DEVICES=0 python pred.py \
# 	--lora_folder ../../ckpt/sst2/${path_after_slash}_sft_f   \
# 	--model_folder ${model_path} \
# 	--output_path ../../data/poison/sst2/${path_after_slash}_sft_f 


# CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
# 	--input_path ../../data/test/Llama-2-7b-hf_sft




