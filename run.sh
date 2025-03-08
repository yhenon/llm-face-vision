#!/bin/bash

# Models and datasets to evaluate
MODELS=(
    "anthropic/claude-3-5-haiku-latest"
    "gemini/gemini-2.0-flash-lite"
    "openai/gpt-4o-mini"

)

DATASETS=(
    "agedb_30"
    "calfw"
    "cplfw"
    "lfw"
)

# Create a new summary_results.csv with headers
echo "Model,$(IFS=,; echo "${DATASETS[*]}")" > summary_results.csv

# Create evaluation_summary.csv with headers (if it doesn't exist)
if [ ! -f evaluation_summary.csv ]; then
    echo "Dataset,Model,Accuracy,Correct,Total" > evaluation_summary.csv
fi

# Process one dataset at a time, but all models in parallel
for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Evaluating dataset: ${dataset}"
    echo "Starting parallel evaluations for all models"
    echo "=========================================="
    
    # Launch all models in parallel for this dataset
    for model in "${MODELS[@]}"; do
        echo "Starting ${model} on ${dataset}"
        python inference.py --model "${model}" --dataset "${dataset}" &
        
        # Store the PID to track it later
        echo "Started process $! for ${model} on ${dataset}"
    done
    
    # Wait for all models to finish processing this dataset
    echo "Waiting for all models to complete evaluations on ${dataset}..."
    wait
    
    echo "Completed all evaluations for dataset: ${dataset}"
    echo "----------------------------------------"
done

echo "All evaluations complete!"
echo "Results are saved in individual CSV files, evaluation_summary.csv, and summary_results.csv"

# Generate a consolidated summary report from evaluation_summary.csv
echo "Generating consolidated summary report..."
python - <<'EOF'
import csv
import pandas as pd

# Read the evaluation summary
df = pd.read_csv('evaluation_summary.csv')

# Pivot the data to get models as rows and datasets as columns
pivot_table = df.pivot_table(index='Model', columns='Dataset', values='Accuracy')

# Format the accuracy values to 4 decimal places
for col in pivot_table.columns:
    pivot_table[col] = pivot_table[col].map('{:.4f}'.format)

# Save to CSV
pivot_table.to_csv('summary_results.csv')
print("Consolidated summary saved to summary_results.csv")
EOF

echo "Done!"