#!/bin/bash

# Define the number of items per job
items_per_job=10
total_items=50
start_index=0
lamb=${1:-0.1}
task=${2:-gsm8k}
poison_data_start=($(seq 0 $((total_items-1))))
# Loop to create jobs in chunks of 5 data points
for (( i=$start_index; i<$total_items; i+=$items_per_job )); do
    # Select the range of data for this job
    chunk=("${poison_data_start[@]:i:items_per_job}")
    # Print the chunk for debugging
    echo "Processing data range: ${chunk[@]}"
    chunk_string=$(printf "%s " "${chunk[@]}")
    # Submit the job for this chunk
    if [[ $task == "agnews" ]]; then
        sbatch virus_agnews.sh "$chunk_string" ${lamb}
    fi
    if [[ $task == "gsm8k" ]]; then
        sbatch virus.sh "$chunk_string" ${lamb}
    fi
    if [[ $task == "sst2" ]]; then
        sbatch virus_sst2.sh "$chunk_string" ${lamb}
    fi
done
