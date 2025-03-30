import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import glob
import sklearn.metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Keep calculate_metrics and calculate_arcface_metrics functions as they are.
# ... (previous code for calculate_metrics and calculate_arcface_metrics) ...
def calculate_metrics(csv_file):
    """
    Calculate precision, recall, accuracy, and refusal rate from a results CSV file
    """
    df = pd.read_csv(csv_file)

    # Check for refusals based on -1 return value in the prediction column
    df['is_refusal'] = df['prediction'] == -1

    # Calculate refusal rate
    refusal_rate = df['is_refusal'].mean() * 100  # as percentage

    # For precision, recall, and accuracy calculations, treat refusals as incorrect predictions
    # Create a new column for adjusted predictions where refusals are set to incorrect (0)
    df['adjusted_prediction'] = df['prediction'].copy()

    # For refusals, set the adjusted prediction to be wrong
    # If ground truth is 1, set prediction to 0; if ground truth is 0, set prediction to 1
    refusal_mask = df['is_refusal']
    df.loc[refusal_mask, 'adjusted_prediction'] = 1 - df.loc[refusal_mask, 'ground_truth']

    # Make sure adjusted_prediction is an integer
    df['adjusted_prediction'] = df['adjusted_prediction'].astype(int)

    # Now calculate precision, recall, and accuracy using the adjusted predictions
    y_true = df['ground_truth']
    y_pred = df['adjusted_prediction']

    # Positive class is 1 (same person)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Extract model name from the filename
    # Handle potential variations in filename structure
    basename = os.path.basename(csv_file)
    parts = basename.split('_')
    if len(parts) > 1:
        model_name = parts[-1].split('.')[0]
        dataset_name = parts[0]
    else: # Fallback if filename doesn't match expected pattern
        model_name = "unknown_model"
        dataset_name = "unknown_dataset"
        print(f"Warning: Could not parse model/dataset from filename: {basename}")


    return {
        'model': model_name,
        'dataset': dataset_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,  # Added accuracy
        'refusal_rate': refusal_rate,
        'total_samples': len(df)
    }

def calculate_arcface_metrics(csv_file):
    """
    Calculate precision-recall curve, accuracy, and F1 scores for ArcFace results
    """
    df = pd.read_csv(csv_file)

    # Ensure we have the required columns
    required_cols = ['is_same', 'similarity']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: ArcFace CSV {csv_file} must contain columns: {required_cols}")
        return None

    # Extract dataset name from file path
    file_name = os.path.basename(csv_file)
    parts = file_name.split('_')
    if len(parts) > 1:
         dataset_name = parts[0]
    else: # Fallback if filename doesn't match expected pattern
        dataset_name = "unknown_dataset"
        print(f"Warning: Could not parse dataset from ArcFace filename: {file_name}")


    # Convert is_same to integer if it's not already
    if df['is_same'].dtype != 'int64':
        try:
             # Handle potential boolean True/False or string 'True'/'False'
             if df['is_same'].dtype == 'bool':
                 df['is_same'] = df['is_same'].astype(int)
             elif df['is_same'].dtype == 'object':
                 df['is_same'] = df['is_same'].str.lower().map({'true': 1, 'false': 0, '1': 1, '0': 0}).astype(int)
             else:
                  df['is_same'] = df['is_same'].astype(int)
        except Exception as e:
            print(f"Error converting 'is_same' column to int in {csv_file}: {e}")
            return None

    y_true = df['is_same']
    scores = df['similarity']

    # Handle potential NaN values
    if y_true.isnull().any() or scores.isnull().any():
        print(f"Warning: NaN values found in {csv_file}. Dropping rows with NaNs.")
        df = df.dropna(subset=['is_same', 'similarity'])
        y_true = df['is_same']
        scores = df['similarity']
        if len(df) == 0:
            print(f"Error: No valid data left in {csv_file} after dropping NaNs.")
            return None


    # Calculate precision-recall curve
    # Ensure y_true contains only 0s and 1s
    if not np.all(np.isin(y_true.unique(), [0, 1])):
        print(f"Error: 'is_same' column in {csv_file} contains values other than 0 or 1: {y_true.unique()}")
        return None

    try:
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true.tolist(), scores.tolist())
    except ValueError as e:
        print(f"Error calculating precision-recall curve for {csv_file}: {e}")
        print(f"  y_true unique values: {y_true.unique()}")
        print(f"  scores sample: {scores.head().tolist()}")
        return None


    # Calculate F1 scores and accuracy for each threshold
    # Handle edge case where thresholds might be empty or only have one value
    if len(thresholds) == 0:
         print(f"Warning: No thresholds generated for {csv_file}. Setting default metrics.")
         # Use average precision and recall if possible, otherwise set default bad values
         avg_precision = sklearn.metrics.average_precision_score(y_true, scores) if len(y_true.unique()) > 1 else 0.0
         # Need a default prediction for other metrics
         y_pred_default = (scores >= scores.mean()).astype(int) if len(scores)>0 else np.zeros_like(y_true)
         best_accuracy = accuracy_score(y_true, y_pred_default)
         best_f1 = f1_score(y_true, y_pred_default, zero_division=0)
         best_acc_threshold = scores.mean() if len(scores)>0 else 0.5
         best_f1_threshold = best_acc_threshold
         best_acc_precision = precision_score(y_true, y_pred_default, zero_division=0)
         best_acc_recall = recall_score(y_true, y_pred_default, zero_division=0)

    else:
        f1_scores = np.zeros(len(thresholds)) # Match length of thresholds
        accuracies = np.zeros(len(thresholds))

        # Note: precision and recall arrays have length len(thresholds) + 1
        # We calculate metrics based on predictions *at* each threshold
        for i, threshold in enumerate(thresholds):
            y_pred = (scores >= threshold).astype(int)
            # Use precision/recall values corresponding *to the next point*
            # as they represent the values *at* or *above* the current threshold
            p = precision[i+1]
            r = recall[i+1]
            if p + r > 0:
                f1_scores[i] = 2 * p * r / (p + r)
            else:
                f1_scores[i] = 0.0 # Set F1 to 0 if P or R is 0

            accuracies[i] = accuracy_score(y_true, y_pred)

        # Find the threshold with the best accuracy
        best_acc_idx = np.argmax(accuracies)
        best_acc_threshold = thresholds[best_acc_idx]
        best_acc_precision = precision[best_acc_idx + 1] # Index matches threshold
        best_acc_recall = recall[best_acc_idx + 1]    # Index matches threshold
        best_accuracy = accuracies[best_acc_idx]

        # Find the threshold with the best F1 score
        best_f1_idx = np.argmax(f1_scores)
        best_f1_threshold = thresholds[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]


    results = {
        'model': 'arcface_r100_v1', # Hardcoded model name for ArcFace
        'dataset': dataset_name,
        'thresholds': thresholds,
        'precision': precision,
        'recall': recall,
        'f1': best_f1, # Best F1 score found
        'best_f1': best_f1,
        'best_f1_threshold': best_f1_threshold,
        'accuracy': best_accuracy, # Best accuracy found
        'best_accuracy': best_accuracy,
        'best_acc_threshold': best_acc_threshold,
        'best_acc_precision': best_acc_precision,
        'best_acc_recall': best_acc_recall,
        'total_samples': len(df) # Add total samples for consistency
    }

    print(f"ArcFace best results for {dataset_name}:")
    print(f"  Best Accuracy: {results['best_accuracy']:.4f} at threshold {results['best_acc_threshold']:.4f}")
    print(f"     Precision: {results['best_acc_precision']:.4f}")
    print(f"     Recall: {results['best_acc_recall']:.4f}")
    print(f"  Best F1: {results['best_f1']:.4f} at threshold {results['best_f1_threshold']:.4f}")
    print(f"  Total Samples: {results['total_samples']}")

    return results

def create_combined_plots(all_metrics, arcface_results=None, output_file=None):
    """
    Create a single figure with multiple subplots for all metrics and datasets
    """
    # Get unique datasets and models
    datasets = sorted(list(set(m['dataset'] for m in all_metrics)), reverse=True) # Keep reverse for layout
    # Ensure models are unique and sorted alphabetically for the legend
    models = sorted(list(set(m['model'] for m in all_metrics)))

    # Define colors and markers for models to ensure consistency across plots
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(models)}
    model_markers = {model: markers[i % len(markers)] for i, model in enumerate(models)}

    # Special color for ArcFace
    arcface_color = 'black'
    arcface_model_name = 'arcface_r100_v1' # Consistent name

    # Create a figure with subplots
    fig = plt.figure(figsize=(18, 12))

    # Create a single legend for the entire figure
    legend_handles = []
    # Add LLM models to legend (alphabetical order)
    for model in models: # Use the alphabetically sorted list
        handle = plt.Line2D([0], [0], marker=model_markers[model], color=model_colors[model],
                           markersize=10, label=model, linestyle='')
        legend_handles.append(handle)

    # Add ArcFace to legend if available
    if arcface_results:
        arcface_handle = plt.Line2D([0], [0], color=arcface_color,
                                  label=arcface_model_name, linestyle='-', linewidth=2)
        legend_handles.append(arcface_handle)

    # --- Row 1: Precision-Recall plots ---
    for i, dataset in enumerate(datasets):
        ax = fig.add_subplot(3, len(datasets), i + 1)
        dataset_metrics = [m for m in all_metrics if m['dataset'] == dataset]

        for metrics in dataset_metrics:
            model = metrics['model']
            # Ensure model exists in our color/marker maps (handles cases if a model appears only in some datasets)
            if model in model_colors:
                 ax.scatter(metrics['recall'], metrics['precision'],
                           s=100, color=model_colors[model], marker=model_markers[model], zorder=3) # zorder=3 to be on top

        # Add ArcFace PR curve
        arcface_pr_plotted = False
        if arcface_results:
            for arc_result in arcface_results:
                if arc_result['dataset'] == dataset:
                    ax.plot(arc_result['recall'], arc_result['precision'],
                           color=arcface_color, linestyle='-', linewidth=2, label=arcface_model_name if i==0 else "", zorder=2) # zorder=2 behind points

                    # Mark the best accuracy point on ArcFace curve
                    ax.scatter(arc_result['best_acc_recall'], arc_result['best_acc_precision'],
                              s=150, color=arcface_color, marker='*',
                              edgecolor='white', linewidth=1.5, zorder=4) # zorder=4 highest
                    arcface_pr_plotted = True

        ax.set_title(f'{dataset.upper()}')
        if i == 0: ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5, zorder=1) # zorder=1 lowest

    # --- Row 2: Accuracy Score plots ---
    for i, dataset in enumerate(datasets):
        ax = fig.add_subplot(3, len(datasets), len(datasets) + i + 1)

        # Filter LLM metrics for this dataset and sort alphabetically
        dataset_llm_metrics = [m for m in all_metrics if m['dataset'] == dataset]
        dataset_llm_metrics = sorted(dataset_llm_metrics, key=lambda x: x['model'])

        # Extract LLM data
        llm_models = [m['model'] for m in dataset_llm_metrics]
        llm_accuracy_scores = [m['accuracy'] for m in dataset_llm_metrics]
        llm_colors = [model_colors[model] for model in llm_models]

        # Prepare combined lists (start with LLMs)
        plot_models = list(llm_models)
        plot_scores = list(llm_accuracy_scores)
        plot_colors = list(llm_colors)
        arcface_acc = None

        # Check if ArcFace results exist for this dataset
        if arcface_results:
            for arc_result in arcface_results:
                if arc_result['dataset'] == dataset:
                    # Prepend ArcFace data to ensure it appears at the top after reversing
                    plot_models.insert(0, arcface_model_name)
                    plot_scores.insert(0, arc_result['best_accuracy'])
                    plot_colors.insert(0, arcface_color)
                    arcface_acc = arc_result['best_accuracy']
                    break

        # Reverse the lists for correct barh plotting order (alphabetical top-down)
        plot_models.reverse()
        plot_scores.reverse()
        plot_colors.reverse()

        # Create horizontal bar chart
        bars = ax.barh(plot_models, plot_scores, color=plot_colors)

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', va='center', ha='left', fontsize=9)

        # Add ArcFace performance line if it exists
        if arcface_acc is not None:
            ax.axvline(x=arcface_acc, color=arcface_color,
                      linestyle='--', alpha=0.7, linewidth=1)

        # Set plot details
        if i == 0:
            ax.set_ylabel('Model')
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('Accuracy')
        ax.set_xlim(0, 1.05)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        # Adjust y-tick labels for the first plot if needed (if labels overlap)
        if i == 0:
             plt.setp(ax.get_yticklabels(), fontsize=10)


    # --- Row 3: Refusal Rate plots ---
    for i, dataset in enumerate(datasets):
        ax = fig.add_subplot(3, len(datasets), 2*len(datasets) + i + 1)

        # Filter metrics for this dataset and sort alphabetically
        dataset_metrics = [m for m in all_metrics if m['dataset'] == dataset]
        dataset_metrics = sorted(dataset_metrics, key=lambda x: x['model'])

        # Extract data
        plot_models = [m['model'] for m in dataset_metrics]
        refusal_rates = [m['refusal_rate'] for m in dataset_metrics]
        plot_colors = [model_colors[model] for model in plot_models] # Use consistent colors

        # Reverse the lists for correct barh plotting order
        plot_models.reverse()
        refusal_rates.reverse()
        plot_colors.reverse()

        # Create horizontal bar chart
        bars = ax.barh(plot_models, refusal_rates, color=plot_colors)

        # Add value labels
        max_rate_for_xlim = max([m['refusal_rate'] for m in all_metrics] + [0]) # Include 0 in case rates are all 0
        label_offset = max_rate_for_xlim * 0.04 if max_rate_for_xlim > 0 else 0.1 # Adjust offset based on max rate
        print(f"Label offset: {label_offset}")
        for bar in bars:
            width = bar.get_width()
            ax.text(width + label_offset, bar.get_y() + bar.get_height()/2,
                   f'{width:.2f}%', va='center', ha='left', fontsize=9)

        # Set plot details
        if i == 0:
            ax.set_ylabel('Model')
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('Refusal Rate (%)')
        # Set xlim based on max refusal rate across all datasets for consistency
        ax.set_xlim(0, max(0.1, max_rate_for_xlim * 1.15)) # Ensure at least 10%, add padding
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        if i == 0:
             plt.setp(ax.get_yticklabels(), fontsize=10)


    # Add the legend at the bottom of the figure
    fig.legend(handles=legend_handles, loc='lower center', ncol=min(len(legend_handles), 6),
               bbox_to_anchor=(0.5, 0), fontsize=12)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.06, 1, 0.96]) # Adjusted rect for legend and title space

    # Add overall title
    plt.suptitle('Model Performance Comparison Across Datasets', fontsize=20, y=0.99) # Slightly lower title

    # Save the figure if output_file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to {output_file}")

    plt.show()


def main():
    # Find all CSV files for LLM results
    csv_files = sorted(glob.glob("out/*.csv"))

    if not csv_files:
        print("No LLM CSV files found in 'out/' directory.")
        # Allow proceeding if only ArcFace files exist
        # return # Removed return to allow plotting only ArcFace if present

    # Calculate metrics for each LLM CSV file
    all_metrics = []
    print("--- Processing LLM Results ---")
    if csv_files:
        for csv_file in csv_files:
            print(f"Processing {csv_file}...")
            try:
                metrics = calculate_metrics(csv_file)
                if metrics: # Check if metrics calculation was successful
                    all_metrics.append(metrics)
                    print(f"  Dataset: {metrics['dataset']}")
                    print(f"  Model: {metrics['model']}")
                    print(f"  Precision: {metrics['precision']:.4f}")
                    print(f"  Recall: {metrics['recall']:.4f}")
                    print(f"  Accuracy: {metrics['accuracy']:.4f}")
                    print(f"  F1 Score: {metrics['f1']:.4f}")
                    print(f"  Refusal Rate: {metrics['refusal_rate']:.2f}%")
                    print(f"  Total Samples: {metrics['total_samples']}")
                else:
                    print(f"  Skipping file due to errors during metric calculation.")
            except Exception as e:
                print(f"  Error processing file {csv_file}: {e}")
            print()
    else:
         print("No LLM result files found.")


    # Process ArcFace results if available
    arcface_results = []
    arcface_files = sorted(glob.glob("arcface/*.csv"))
    print("\n--- Processing ArcFace Results ---")
    if arcface_files:
        for arcface_file in arcface_files:
            print(f"Processing ArcFace file: {arcface_file}...")
            try:
                arc_metrics = calculate_arcface_metrics(arcface_file)
                if arc_metrics: # Check if calculation was successful
                    arcface_results.append(arc_metrics)
                else:
                     print(f"  Skipping ArcFace file due to errors during metric calculation.")
            except Exception as e:
                 print(f"  Error processing ArcFace file {arcface_file}: {e}")
            print() # Print blank line after each file processing message
    else:
        print("No ArcFace result files found in 'arcface/' directory.")

    # Create the combined plot if there's any data to plot
    if all_metrics or arcface_results:
         print("\n--- Generating Combined Plot ---")
         # Ensure datasets/models derived correctly even if only one type of result exists
         if not all_metrics and arcface_results:
              # If only ArcFace, create dummy all_metrics for structure (won't be plotted)
              all_metrics = [{'dataset': r['dataset'], 'model': 'dummy_llm', 'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0, 'refusal_rate': 0, 'total_samples': 0} for r in arcface_results]
              print("Note: Only ArcFace results found. Plotting ArcFace curves and accuracy.")


         create_combined_plots(all_metrics, arcface_results, "combined_performance_metrics.png")
    else:
         print("\nNo data found from either LLM or ArcFace results. Skipping plot generation.")


if __name__ == "__main__":
    main()