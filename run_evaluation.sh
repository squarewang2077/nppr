#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# Number of runs for each combination
NUM_RUNS=10

for model in resnet18 resnet50 wide_resnet50_2 vgg16
do
    for dataset in cifar10 cifar100 tinyimagenet
    do
        for run in $(seq 1 $NUM_RUNS)
        do
            echo "=========================================="
            echo "Running: Model=$model, Dataset=$dataset, Run=$run/$NUM_RUNS"
            echo "=========================================="

            python test.py \
                --dataset $dataset \
                --arch $model \
                --clf_ckpt ./model_zoo/trained_model/${model}_${dataset}.pth \
                --epsilon 0.062 \
                --norm_type linf \
                --num_samples 100 \
                --attack_steps 20 \
                --step_size 0.00784 \
                --max_batches 10 \
                --log_dir ./logs/${model}_${dataset}/run_${run}

            echo "Completed run $run for $model on $dataset"
            echo ""
        done
    done
done

echo "All evaluations completed!"
