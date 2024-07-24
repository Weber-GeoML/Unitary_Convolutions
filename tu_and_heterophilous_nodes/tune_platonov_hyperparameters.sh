#!/bin/bash

# Define the lists for each parameter
layer_types=("Unitary")
num_layers_list=(5 10 15)
learning_rates=(0.001 0.0001 0.00001)
dropouts=(0.1 0.2 0.3)

# Nested loops to iterate over each combination of parameters
for layer_type in "${layer_types[@]}"; do
    for num_layers in "${num_layers_list[@]}"; do
        for learning_rate in "${learning_rates[@]}"; do
            for dropout in "${dropouts[@]}"; do
                # Construct the command
                cmd="python run_node_classification.py --layer_type $layer_type --num_layers $num_layers --learning_rate $learning_rate --dropout $dropout"
                
                # Print and execute the command
                echo "Running: $cmd"
                $cmd
            done
        done
    done
done
