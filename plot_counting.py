import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import matplotlib.gridspec as gridspec

def analyze_face_count_csvs():
    # Find all CSV files in the out_counting folder
    csv_files = glob('out_counting/*.csv')
    
    if not csv_files:
        print("No CSV files found in the out_counting folder.")
        return
    
    results = []
    
    # Process each CSV file
    for csv_file in csv_files:
        model_name = os.path.basename(csv_file).replace('.csv', '').replace('face_counting_', '')
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Count total samples and refusals
            total_samples = len(df)
            refusals = (df['response_num_faces'] == -1).sum()
            refusal_rate = refusals / total_samples if total_samples > 0 else 0
            
            # Filter out refusals for accuracy calculation details
            valid_df = df[df['response_num_faces'] != -1].copy()
            
            # Calculate percentage error and absolute error for each valid prediction
            if not valid_df.empty:
                valid_df['perc_error'] = 100 * abs(valid_df['gt_num_faces'] - valid_df['response_num_faces']) / valid_df['gt_num_faces']
                valid_df['abs_error'] = abs(valid_df['gt_num_faces'] - valid_df['response_num_faces'])
                
                # Calculate MAE (Mean Absolute Error)
                mae = valid_df['abs_error'].mean()
                
                # Calculate MAE including refusals (count refusals as a guess of 0 faces)
                # Create a temporary dataframe with refusals to calculate their error correctly
                refusal_df = df[df['response_num_faces'] == -1].copy()
                refusal_error = refusal_df['gt_num_faces'].sum() if not refusal_df.empty else 0
                mae_with_refusals = (valid_df['abs_error'].sum() + refusal_error) / total_samples
                
                # Count correct predictions at each threshold
                correct_10_percent = (valid_df['perc_error'] <= 10).sum()
                correct_25_percent = (valid_df['perc_error'] <= 25).sum()
                correct_50_percent = (valid_df['perc_error'] <= 50).sum()
            else:
                mae = float('nan')
                mae_with_refusals = float('nan')
                correct_10_percent = 0
                correct_25_percent = 0
                correct_50_percent = 0
            
            # Calculate accuracy percentages including refusals as incorrect
            within_10_percent = (correct_10_percent / total_samples) * 100
            within_25_percent = (correct_25_percent / total_samples) * 100
            within_50_percent = (correct_50_percent / total_samples) * 100
            
            results.append({
                'model': model_name,
                'refusal_rate': refusal_rate * 100,  # Convert to percentage
                'within_10_percent': within_10_percent,
                'within_25_percent': within_25_percent,
                'within_50_percent': within_50_percent,
                'num_samples': total_samples,
                'valid_samples': len(valid_df),
                'mae': mae,
                'mae_with_refusals': mae_with_refusals
            })
            
            print(f"Processed {model_name}:")
            print(f"  MAE (valid samples only): {mae:.2f}")
            print(f"  MAE (counting refusals): {mae_with_refusals:.2f}")
            print(f"  Refusal rate: {refusal_rate:.2%}")
            print(f"  Accuracy within 10% (counting refusals as wrong): {within_10_percent:.2f}%")
            print(f"  Accuracy within 25% (counting refusals as wrong): {within_25_percent:.2f}%")
            print(f"  Accuracy within 50% (counting refusals as wrong): {within_50_percent:.2f}%")
            print(f"  Samples: {total_samples} (valid: {len(valid_df)}, refusals: {refusals})")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("No valid data found to plot.")
        return
    
    # Sort by MAE (ascending)
    results_df = results_df.sort_values('mae_with_refusals')
    
    # Create plot
    create_combined_plot(results_df)



def create_combined_plot(results_df):
    # Create figure with a grid layout - now with 4 subplots
    fig = plt.figure(figsize=(16, 24))  # Increased height for additional plots
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.3)
    
    # Extract model names for consistent ordering
    models = results_df['model'].tolist()
    
    # Define consistent colors for models
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(models)}
    
    # 1. Create accuracy metrics subplot - grouped by accuracy level
    ax1 = plt.subplot(gs[0])
    
    # Prepare data for grouped by accuracy level
    accuracy_data = {
        'Within 10%': results_df['within_10_percent'].tolist(),
        'Within 25%': results_df['within_25_percent'].tolist(),
        'Within 50%': results_df['within_50_percent'].tolist()
    }
    
    # Number of accuracy metrics and models
    n_metrics = len(accuracy_data)
    n_models = len(models)
    
    # Set up x positions
    indices = np.arange(n_metrics)
    width = 0.8 / n_models  # Adjust bar width based on number of models
    
    # Plot bars grouped by accuracy level
    for i, model in enumerate(models):
        offset = (i - n_models / 2 + 0.5) * width
        model_values = [results_df.loc[results_df['model'] == model, f'within_{level}_percent'].values[0] 
                        for level in ['10', '25', '50']]
        
        bars = ax1.bar(indices + offset, model_values, width, label=model, color=model_colors[model])
        
        # Add data labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if value is positive
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Set labels and title
    ax1.set_xlabel('Accuracy Level')
    ax1.set_ylabel('Percentage of Samples (%)')
    ax1.set_title('Face Detection Accuracy Comparison (Refusals Counted as Incorrect)')
    ax1.set_xticks(indices)
    ax1.set_xticklabels(list(accuracy_data.keys()))
    
    # Add a legend with smaller font
    ax1.legend(loc='upper left', fontsize=9)
    
    # 2. Create refusal rates subplot
    ax2 = plt.subplot(gs[3])
    
    # Plot refusal rates using the same model order and MATCHING COLORS
    refusal_values = [results_df.loc[results_df['model'] == model, 'refusal_rate'].values[0] for model in models]
    bars = ax2.bar(models, refusal_values, color=[model_colors[model] for model in models])
    
    # Add data labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Set labels and title
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Refusal Rate (%)')
    ax2.set_title('Face Detection Refusal Rates')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Create MAE subplot (without refusals)
    ax3 = plt.subplot(gs[1])
    
    # Plot MAE values (without refusals)
    mae_values = [results_df.loc[results_df['model'] == model, 'mae'].values[0] for model in models]
    mae_bars = ax3.bar(np.arange(len(models)), mae_values, 
                      color=[model_colors[model] for model in models])
    
    # Add data labels on bars
    for bar in mae_bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Set labels and title
    ax3.set_ylabel('Mean Absolute Error (MAE)')
    ax3.set_title('Face Detection Mean Absolute Error - Valid Samples Only')
    ax3.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    
    # Add a grid for better readability
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. Create MAE subplot (with refusals)
    ax4 = plt.subplot(gs[2])
    
    # Plot MAE values (with refusals)
    mae_with_refusals_values = [results_df.loc[results_df['model'] == model, 'mae_with_refusals'].values[0] for model in models]
    mae_refusal_bars = ax4.bar(np.arange(len(models)), mae_with_refusals_values, 
                              color=[model_colors[model] for model in models])
    
    # Add data labels on bars
    for bar in mae_refusal_bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Set labels and title
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Mean Absolute Error (MAE)')
    ax4.set_title('Face Detection Mean Absolute Error - Including Refusals')
    #ax4.set_xticks(np.arange(len(models)))
    #ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    
    # Add a grid for better readability
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add padding to avoid y-axis label overlap
    plt.tight_layout()
    # Additional padding on the left side to prevent y-axis label overlap
    fig.subplots_adjust(left=0.1)
    
    # Save the plot
    plt.savefig('face_detection_performance_comprehensive.png', dpi=300)
    print("Comprehensive plot saved as face_detection_performance_comprehensive.png")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    analyze_face_count_csvs()