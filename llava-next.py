import time
import argparse
import os
import cv2
import csv
from tqdm import tqdm
import torch
from PIL import Image
import copy
import numpy as np

# Import LLaVA-next specific modules
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates

def setup_llava_model(model_path="lmms-lab/llava-onevision-qwen2-7b-si", model_name=None, device="cuda"):
    """Initialize the LLaVA model and required components."""
    if model_name is None:
        model_name = get_model_name_from_path(model_path)
    
    llava_model_args = {
        "multimodal": True,
        "attn_implementation": "sdpa",
    }
    
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path, None, model_name, device_map=device, **llava_model_args
    )
    model.eval()
    return tokenizer, model, image_processor

def prepare_images_for_llava(img1, img2, image_processor, model_config):
    """Prepare two images for LLaVA model input by concatenating them horizontally."""
    # Ensure both images have the same height
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Resize if heights don't match (keeping aspect ratio)
    if h1 != h2:
        target_height = min(h1, h2)
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2
        
        new_w1 = int(target_height * aspect_ratio1)
        new_w2 = int(target_height * aspect_ratio2)
        
        img1 = cv2.resize(img1, (new_w1, target_height))
        img2 = cv2.resize(img2, (new_w2, target_height))
    
    # Add a small separator between images (a vertical white line)
    separator_width = 5
    separator = np.ones((img1.shape[0], separator_width, 3), dtype=np.uint8) * 255
    
    # Concatenate images horizontally with separator
    combined_img = cv2.hconcat([img1, separator, img2])
    
    # Convert to PIL
    pil_combined = Image.fromarray(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
    
    # Process the combined image
    image_tensor = process_images([pil_combined], image_processor, model_config)[0]
    
    return [image_tensor]

def generate_llava_response(tokenizer, model, image_tensors, device="cuda"):
    """Generate a response from LLaVA model for the face matching task."""
    # Use appropriate conversation template (adjust based on your LLaVA model)
    conv_template = "qwen_1_5"  # Common for many LLaVA models, change if needed
    
    # Create the prompt with a single image token for our concatenated image
    question = (
        f"{DEFAULT_IMAGE_TOKEN}\n"
        "The image contains a face on the left and a face on the right. Are these two faces of the same person? Respond only with YES or NO. Any answer other than YES or NO will be considered a failure."
    )
    
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Tokenize the prompt
    input_ids = tokenizer_image_token(
        prompt, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors="pt"
    ).unsqueeze(0).to(device)
    
    # The image has already been concatenated, so we just need to use it
    # Add batch dimension if not already present
    if len(image_tensors[0].shape) == 3:
        image_tensor = image_tensors[0].unsqueeze(0)
    else:
        image_tensor = image_tensors[0]
    
    # Estimate reasonable image size (can be approximate)
    image_sizes = [(448, 224)]  # Width is roughly double height since we concatenated horizontally
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=20,
        )
    
    # Decode output
    response = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return response

def parse_llava_response(response):
    """Parse the LLaVA model's response to get the binary prediction."""
    # Look for "0" or "1" in the response
    if response.lower() == "yes":
        return 1
    elif response.lower() == 'no':
        return 0
    else:
        return -1

def evaluate_model(model_path, model_name, subdir, dataset, device="cuda"):
    """
    Evaluate a LLaVA model on a face matching dataset and save results to CSV
    
    Args:
        model_path (str): Path or HF repo of the LLaVA model
        model_name (str): Name of the model architecture
        subdir (str): Directory containing the dataset
        dataset (str): Dataset name
        device (str): Device to run inference on
    """
    # Create a model short name for the filename
    model_short_name = model_path.split('/')[-1]
    csv_filename = os.path.join("out", f"{dataset}_{model_short_name}.csv")
    
    print(f"Evaluating {model_path} on {dataset} dataset...")
    
    # Initialize LLaVA model
    tokenizer, model, image_processor = setup_llava_model(model_path, model_name, device)
    
    # Read the annotation file
    annotation_path = os.path.join(subdir, f"{dataset}_ann.txt")
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()
    
    # Set up the CSV file and writer
    fieldnames = ['image1', 'image2', 'ground_truth', 'prediction', 'correct', 'raw_response']
    
    # Check if the file exists to determine if we need to write a header
    file_exists = os.path.isfile(csv_filename)
    
    # Open the file in append mode so we can add rows incrementally
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
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
            
            try:
                # Process images for LLaVA
                image_tensors = prepare_images_for_llava(img1, img2, image_processor, model.config)
                image_tensors = [img.to(dtype=torch.float16, device=device) for img in image_tensors]
                
                # Get response from LLaVA model
                raw_response = generate_llava_response(tokenizer, model, image_tensors, device)
                parsed_response = parse_llava_response(raw_response)
                
                # Create result dictionary
                result = {
                    'image1': img_url1,
                    'image2': img_url2,
                    'ground_truth': is_match,
                    'prediction': parsed_response,
                    'correct': parsed_response == is_match,
                    'raw_response': raw_response.strip().replace('\n', ' ').replace('\r', ' ')
                }
                
                # Update statistics
                if result['correct']:
                    correct_count += 1
                total_count += 1
                
                # Immediately write this result to the CSV file
                writer.writerow(result)
                # Flush to make sure it's written to disk
                csv_file.flush()
                
            except Exception as e:
                print(f"Error processing {img_url1} and {img_url2}: {str(e)}")
    
    finally:
        # Close the CSV file
        csv_file.close()
    
    # Calculate and print accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset}")
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    print(f"Results saved to {csv_filename}")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLaVA-next on image comparison tasks')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Model path or HF repo (e.g., "lmms-lab/llava-onevision-qwen2-7b-si")')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model architecture name (optional, will be inferred if not provided)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., "agedb_30", "calfw", "cplfw", "lfw")')
    parser.add_argument('--base_dir', type=str, default='./val',
                        help='Base directory containing datasets (default: ./val)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (default: cuda)')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_model(args.model_path, args.model_name, args.base_dir, args.dataset, args.device)

if __name__ == "__main__":
    main()
