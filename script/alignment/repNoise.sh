#!/bin/bash
#SBATCH -J repnoise                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=30G
#SBATCH -o repnoise_0-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences


module load anaconda3/2023.03
module load cuda/11.8.0

source activate hts
alpha=0.1
beta=0.001
model_path=${1:-meta-llama/Llama-2-7b-hf}      
path_after_slash=$(basename "$model_path") 
echo "alpha is: $alpha"
echo "beta is: $beta"
echo "The model path is: $model_path"
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory

# Has to lower batch size due to oom
CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path}  \
	--data_path PKU-Alignment/BeaverTails_safe \
	--bf16 True \
	--output_dir ckpt/${path_after_slash}_repnoise_${alpha}_${beta} \
	--num_train_epochs 20 \
	--per_device_train_batch_size 5 \
	--per_device_eval_batch_size 5 \
	--gradient_accumulation_steps 1 \
	--evaluation_strategy "no" \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate  5e-4\
	--weight_decay 0.1 \
	--warmup_ratio 0\
	--lr_scheduler_type "constant" \
	--logging_steps 1 \
	--tf32 True \
	--cache_dir cache \
	--optimizer rep_noise \
	--sample_num 5000 \
	--bad_sample_num 5000 \
	--lamb ${beta} \
	--rho ${alpha}

cd poison/evaluation  

# CUDA_VISIBLE_DEVICES=0 python pred.py \
# 	--lora_folder ../../ckpt/${path_after_slash}_repnoise_${alpha}_${beta} \
# 	--model_folder ${model_path} \
# 	--output_path ../../data/poison/${path_after_slash}_repnoise_${alpha}_${beta}

# CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
# 	--input_path ../../data/poison/${path_after_slash}_repnoise_${alpha}_${beta}

# cd ../../gsm8k

# CUDA_VISIBLE_DEVICES=0 python pred_eval.py   \
# 	--lora_folder ../ckpt/${path_after_slash}_repnoise_${alpha}_${beta} \
# 	--model_folder ${model_path} \
# 	--output_path ../data/gsm8k/${path_after_slash}_repnoise_${alpha}_${beta}