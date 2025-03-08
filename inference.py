import time
import argparse
from litellm import completion
import os
import cv2
from utils import prepare_images_for_llm, parse_response
import csv
from tqdm import tqdm

def evaluate_model(model_name, subdir, dataset):
    """
    Evaluate a single model on a single dataset and save results to CSV
    
    Args:
        model_name (str): The model identifier (e.g., "openai/gpt-4o-mini")
        subdir (str): Directory containing the dataset
        dataset (str): Dataset name
    """
    # Extract model name without provider prefix for filename
    model_short_name = model_name.split('/')[-1]
    csv_filename = os.path.join("out", f"{dataset}_{model_short_name}.csv")
    
    print(f"Evaluating {model_name} on {dataset} dataset...")
    
    # Read the annotation file
    annotation_path = os.path.join(subdir, f"{dataset}_ann.txt")
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()
    
    # Set up the CSV file and writer
    fieldnames = ['image1', 'image2', 'ground_truth', 'prediction', 'correct', 'raw_response']
    
    # Check if the file exists to determine if we need to write a header
    file_exists = os.path.isfile(csv_filename)
    
    # Open the file in append mode so we can add rows incrementally
    csv_file = open(csv_filename, 'a', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write header only if file is new
    if not file_exists:
        writer.writeheader()
    
    # If file exists, determine how many pairs we've already processed
    processed_pairs = set()
    if file_exists:
        with open(csv_filename, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pair_key = f"{row['image1']}_{row['image2']}"
                processed_pairs.add(pair_key)
    
    # Track results for final statistics
    correct_count = 0
    total_count = 0
    
    try:
        # Process each image pair
        for line in tqdm(annotations):
            is_match, img_url1, img_url2 = line.strip().split(' ')
            is_match = int(is_match)
            
            # Create a unique key for this pair
            pair_key = f"{img_url1}_{img_url2}"
            
            # Skip if this pair was already processed
            if pair_key in processed_pairs:
                continue
            
            # Load images
            img1 = cv2.imread(os.path.join(subdir, img_url1))
            img2 = cv2.imread(os.path.join(subdir, img_url2))
            
            # Skip if image loading failed
            if img1 is None or img2 is None:
                print(f"Warning: Failed to load images {img_url1} or {img_url2}, skipping")
                continue
                
            # Prepare prompt
            prompt = prepare_images_for_llm(img1, img2)
            
            try:
                # Make API call
                response = completion(model=model_name, messages=prompt, temperature=0.0, max_tokens=5)
                parsed_response = parse_response(response.choices[0].message.content)
                
                # Create result dictionary
                result = {
                    'image1': img_url1,
                    'image2': img_url2,
                    'ground_truth': is_match,
                    'prediction': parsed_response,
                    'correct': parsed_response == is_match,
                    'raw_response': response.choices[0].message.content.strip().replace('\n', ' ').replace('\r', ' ')
                }
                
                # Update statistics
                if result['correct']:
                    correct_count += 1
                total_count += 1
                
                # Immediately write this result to the CSV file
                writer.writerow(result)
                # Flush to make sure it's written to disk
                csv_file.flush()
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing {img_url1} and {img_url2}: {str(e)}")
    
    finally:
        # Close the CSV file
        csv_file.close()
    
    # Count results from previously processed pairs if the file existed
    if file_exists:
        with open(csv_filename, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pair_key = f"{row['image1']}_{row['image2']}"
                if pair_key not in processed_pairs:  # Skip pairs we just processed
                    if row['correct'].lower() == 'true':
                        correct_count += 1
                    total_count += 1
    
    # Calculate and print accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    print(f"Results saved to {csv_filename}")
    
    
    return accuracy

def load_api_keys():
    """Load API keys from files"""
    # Load OpenAI API key if file exists
    if os.path.exists("keys/openai_key.txt"):
        with open("keys/openai_key.txt", "r") as f:
            os.environ["OPENAI_API_KEY"] = f.read().strip()
    
    # Load Claude API key if file exists
    if os.path.exists("keys/claude_key.txt"):
        with open("keys/claude_key.txt", "r") as f:
            os.environ["ANTHROPIC_API_KEY"] = f.read().strip()
    
    # Load Gemini API key if file exists
    if os.path.exists("keys/gemini_key.txt"):
        with open("keys/gemini_key.txt", "r") as f:
            os.environ["GEMINI_API_KEY"] = f.read().strip()

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLMs on image comparison tasks')
    parser.add_argument('--model', type=str, required=True, 
                        help='Model identifier (e.g., "openai/gpt-4o-mini")')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., "agedb_30", "calfw", "cplfw", "lfw")')
    parser.add_argument('--base_dir', type=str, default='./val',
                        help='Base directory containing datasets (default: ./val)')
    
    args = parser.parse_args()
    
    # Load API keys
    load_api_keys()
    
    # Run evaluation
    evaluate_model(args.model, args.base_dir, args.dataset)

if __name__ == "__main__":
    main()