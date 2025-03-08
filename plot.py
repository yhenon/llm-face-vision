import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import sklearn.metrics
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(csv_file):
    """
    Calculate precision, recall, and refusal rate from a results CSV file
    """
    df = pd.read_csv(csv_file)
    
    # Check for refusals based on -1 return value in the prediction column
    df['is_refusal'] = df['prediction'] == -1
    
    # Calculate refusal rate
    refusal_rate = df['is_refusal'].mean() * 100  # as percentage
    
    # For precision and recall calculations, treat refusals as incorrect predictions
    # Create a new column for adjusted predictions where refusals are set to incorrect (0)
    df['adjusted_prediction'] = df['prediction'].copy()
    
    # For refusals, set the adjusted prediction to be wrong
    # If ground truth is 1, set prediction to 0; if ground truth is 0, set prediction to 1
    refusal_mask = df['is_refusal']
    df.loc[refusal_mask, 'adjusted_prediction'] = 1 - df.loc[refusal_mask, 'ground_truth']
    
    # Make sure adjusted_prediction is an integer
    df['adjusted_prediction'] = df['adjusted_prediction'].astype(int)
    
    # Now calculate precision and recall using the adjusted predictions
    y_true = df['ground_truth']
    y_pred = df['adjusted_prediction']
    
    # Positive class is 1 (same person)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Extract model name from the filename
    model_name = os.path.basename(csv_file).split('_')[-1].split('.')[0]
    
    # Extract dataset name from the filename
    dataset_name = os.path.basename(csv_file).split('_')[0]
    
    return {
        'model': model_name,
        'dataset': dataset_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'refusal_rate': refusal_rate,
        'total_samples': len(df)
    }

def calculate_arcface_metrics(csv_file):
    """
    Calculate precision-recall curve and F1 scores for ArcFace results
    """
    df = pd.read_csv(csv_file)
    
    # Ensure we have the required columns
    required_cols = ['is_same', 'similarity']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: ArcFace CSV must contain columns: {required_cols}")
        return None
    
    # Extract dataset name from file path
    file_name = os.path.basename(csv_file)
    dataset_name = file_name.split('_')[0]
    
    # Convert is_same to integer if it's not already
    if df['is_same'].dtype != 'int64':
        df['is_same'] = df['is_same'].astype(int)
    
    y_true = df['is_same']
    scores = df['similarity']
    
    # Calculate precision-recall curve
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true.tolist(), scores.tolist())
    
    # Calculate F1 scores for each threshold
    f1_scores = np.zeros_like(thresholds)
    for i, (p, r) in enumerate(zip(precision[:-1], recall[:-1])):
        if p + r > 0:  # Avoid division by zero
            f1_scores[i] = 2 * p * r / (p + r)
    
    # Find the threshold with the best F1 score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_f1 = f1_scores[best_idx]
    
    results = {
        'model': 'arcface',
        'dataset': dataset_name,
        'thresholds': thresholds,
        'precision': precision,
        'recall': recall,
        'f1': best_f1,
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'best_precision': best_precision,
        'best_recall': best_recall
    }
    
    print(f"ArcFace best results for {dataset_name}:")
    print(f"  Best F1: {results['best_f1']:.4f} at threshold {results['best_threshold']:.4f}")
    print(f"  Precision: {results['best_precision']:.4f}")
    print(f"  Recall: {results['best_recall']:.4f}")
    
    return results

def create_combined_plots(all_metrics, arcface_results=None, output_file=None):
    """
    Create a single figure with multiple subplots for all metrics and datasets
    """
    # Get unique datasets and models
    datasets = sorted(list(set(m['dataset'] for m in all_metrics)))
    models = sorted(list(set(m['model'] for m in all_metrics)))
    
    # Define colors and markers for models to ensure consistency across plots
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(models)}
    model_markers = {model: markers[i % len(markers)] for i, model in enumerate(models)}
    
    # Special color for ArcFace
    arcface_color = 'black'
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # We'll create 3 rows of subplots:
    # 1. Precision-Recall
    # 2. F1 Scores
    # 3. Refusal Rates
    
    # Create a single legend for the entire figure
    legend_handles = []
    for model in models:
        handle = plt.Line2D([0], [0], marker=model_markers[model], color=model_colors[model], 
                           markersize=10, label=model, linestyle='')
        legend_handles.append(handle)
    
    # Add ArcFace to legend if available
    if arcface_results:
        arcface_handle = plt.Line2D([0], [0], color=arcface_color, 
                                  label='arcface_r100_v1', linestyle='-', linewidth=2)
        legend_handles.append(arcface_handle)
    
    # Row 1: Precision-Recall plots
    for i, dataset in enumerate(datasets):
        ax = fig.add_subplot(3, len(datasets), i + 1)
        
        # Filter metrics for this dataset
        dataset_metrics = [m for m in all_metrics if m['dataset'] == dataset]
        
        # Plot each model as a point
        for metrics in dataset_metrics:
            model = metrics['model']
            ax.scatter(metrics['recall'], metrics['precision'], 
                      s=100, color=model_colors[model], marker=model_markers[model])
        
        # Add ArcFace precision-recall curve if available
        if arcface_results:
            for arc_result in arcface_results:
                if arc_result['dataset'] == dataset:
                    ax.plot(arc_result['recall'], arc_result['precision'], 
                           color=arcface_color, linestyle='-', linewidth=2)
                    
                    # Mark the best F1 point
                    ax.scatter(arc_result['best_recall'], arc_result['best_precision'], 
                              s=150, color=arcface_color, marker='*', 
                              edgecolor='white', linewidth=1.5)
        
        # Set plot details
        ax.set_title(f'{dataset.upper()}')
        if i == 0:  # Only add y-label for the first plot in the row
            ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Add diagonal line representing random performance
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5)
    
    # Row 2: F1 Score plots
    for i, dataset in enumerate(datasets):
        ax = fig.add_subplot(3, len(datasets), len(datasets) + i + 1)
        
        # Filter metrics for this dataset
        dataset_metrics = [m for m in all_metrics if m['dataset'] == dataset]
        
        # Sort metrics by model name for consistency with other plots
        dataset_metrics = sorted(dataset_metrics, key=lambda x: x['model'])
        
        # Extract data for plotting
        plot_models = [m['model'] for m in dataset_metrics]
        f1_scores = [m['f1'] for m in dataset_metrics]
        
        # Create horizontal bar chart with consistent colors
        bars = ax.barh(plot_models, f1_scores, color=[model_colors[model] for model in plot_models])
        
        # Add ArcFace F1 score if available
        if arcface_results:
            for arc_result in arcface_results:
                if arc_result['dataset'] == dataset:
                    # Add ArcFace as a separate additional bar with proper positioning
                    # Add ArcFace to the top of the plot
                    all_models = ['arcface_r100_v1'] + plot_models
                    
                    # Clear existing plot and redraw with ArcFace included
                    ax.clear()
                    all_f1_scores = [arc_result['best_f1']] + f1_scores
                    all_colors = [arcface_color] + [model_colors[model] for model in plot_models]
                    
                    # Create horizontal bar chart with all models
                    bars = ax.barh(all_models, all_f1_scores, color=all_colors)
                    
                    # Add value labels
                    for j, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                               f'{width:.3f}', va='center')
                    
                    # Add a horizontal line across the plot to highlight ArcFace performance
                    ax.axvline(x=arc_result['best_f1'], color=arcface_color, 
                              linestyle='--', alpha=0.7, linewidth=1)
                else:
                    # If there's no ArcFace result for this dataset, add value labels to the original bars
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                               f'{width:.3f}', va='center')
        else:
            # If there are no ArcFace results at all, add value labels to the original bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', va='center')
        
        # Set plot details
        if i == 0:  # Only add y-label and tick labels for the first plot in the row
            ax.set_ylabel('Model')
        else:
            ax.set_yticklabels([])  # Hide y-tick labels for all but the first plot
        ax.set_xlabel('F1 Score')
        ax.set_xlim(0, 1.05)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Row 3: Refusal Rate plots
    for i, dataset in enumerate(datasets):
        ax = fig.add_subplot(3, len(datasets), 2*len(datasets) + i + 1)
        
        # Filter metrics for this dataset
        dataset_metrics = [m for m in all_metrics if m['dataset'] == dataset]
        
        # Sort metrics by model name for consistency with other plots
        dataset_metrics = sorted(dataset_metrics, key=lambda x: x['model'])
        
        # Extract data for plotting
        plot_models = [m['model'] for m in dataset_metrics]
        refusal_rates = [m['refusal_rate'] for m in dataset_metrics]
        
        # Create horizontal bar chart with consistent colors
        bars = ax.barh(plot_models, refusal_rates, color=[model_colors[model] for model in plot_models])
        
        # Add value labels to all bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}%', va='center')
        
        # Set plot details
        if i == 0:  # Only add y-label and tick labels for the first plot in the row
            ax.set_ylabel('Model')
        else:
            ax.set_yticklabels([])  # Hide y-tick labels for all but the first plot
        ax.set_xlabel('Refusal Rate (%)')
        max_rate = max([m['refusal_rate'] for m in all_metrics]) if all_metrics else 10
        ax.set_xlim(0, max_rate * 1.2)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add row titles
    #row_titles = ['Precision vs. Recall', 'F1 Scores', 'Refusal Rates']
    #for i, title in enumerate(row_titles):
    #    fig.text(0.5, 0.99 - i*0.33, title, ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Add the legend at the bottom of the figure
    fig.legend(handles=legend_handles, loc='lower center', ncol=min(len(models) + 1, 6), 
               bbox_to_anchor=(0.5, 0), fontsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space for the legend at the bottom
    
    # Add overall title
    plt.suptitle('Model Performance Comparison Across Datasets', fontsize=20, y=1.02)
    
    # Save the figure if output_file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to {output_file}")
    
    plt.show()

def main():
    # Find all CSV files
    csv_files = sorted(glob.glob("out/*.csv"))
    
    if not csv_files:
        print("No CSV files found")
        return
    
    # Calculate metrics for each CSV file
    all_metrics = []
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        metrics = calculate_metrics(csv_file)
        all_metrics.append(metrics)
        print(f"  Dataset: {metrics['dataset']}")
        print(f"  Model: {metrics['model']}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  Refusal Rate: {metrics['refusal_rate']:.2f}%")
        print(f"  Total Samples: {metrics['total_samples']}")
        print()
    
    # Process ArcFace results if available
    arcface_results = []
    arcface_files = sorted(glob.glob("arcface/*.csv"))
    
    for arcface_file in arcface_files:
        print(f"Processing ArcFace file: {arcface_file}...")
        arc_metrics = calculate_arcface_metrics(arcface_file)
        if arc_metrics:
            arcface_results.append(arc_metrics)
    
    # Create the combined plot
    create_combined_plots(all_metrics, arcface_results, "combined_performance_metrics.png")

if __name__ == "__main__":
    main()