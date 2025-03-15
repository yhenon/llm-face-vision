import time
import argparse
from litellm import completion
import os
import cv2
from PIL import Image
import io
import csv
from tqdm import tqdm
import base64

def parse_widerface_annotations(annotation_file):
    """
    Parse the WiderFace annotation file and return a dictionary mapping
    image filenames to the number of faces.
    
    Args:
        annotation_file (str): Path to the WiderFace annotation file
        
    Returns:
        dict: Dictionary mapping image filenames to number of faces
    """
    annotations = {}
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        # Get image filename
        filename = lines[i].strip()
        i += 1
        
        # Get number of faces
        num_faces = int(lines[i].strip())
        i += 1
        
        # Check if this is a "crowd" annotation (indicated by count of 1 and specific bbox format)
        is_crowd = False
        if num_faces == 1:
            bbox_line = lines[i].strip()
            # Check if this is the special "crowd" annotation format (all zeros or similar pattern)
            if bbox_line == "0 0 0 0 0 0 0 0 0 0":
                is_crowd = True
                print(f"Skipping crowd image: {filename}")
        
        # Skip bounding box lines
        i += num_faces
        
        # Store in dictionary only if not a crowd image
        if not is_crowd:
            annotations[filename] = num_faces
        
    return annotations

def parse_face_count_response(response_text):
    """
    Parse the model's response to extract the number of faces.
    
    Args:
        response_text (str): The model's raw response
        
    Returns:
        int: The parsed face count, or -1 if parsing failed
    """
    try:
        # Lower case and remove extra spaces
        text = response_text.lower().strip()
        
        # Try to find a number in the response
        import re
        numbers = re.findall(r'\b(\d+)\b', text)
        
        if numbers:
            # Return the first number found
            return int(numbers[0])
        return -1
        
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        return -1
    
def evaluate_face_counter(model_name, dataset_dir, annotation_file):
    """
    Evaluate a model on the face counting task and save results to CSV
    
    Args:
        model_name (str): The model identifier (e.g., "openai/gpt-4o")
        dataset_dir (str): Directory containing the images
        annotation_file (str): Path to the WiderFace annotation file
    """
    # Extract model name without provider prefix for filename
    model_short_name = model_name.split('/')[-1]
    csv_filename = os.path.join("out_counting", f"face_counting_{model_short_name}.csv")
    
    print(f"Evaluating {model_name} on face counting task...")
    
    # Parse annotations
    annotations = parse_widerface_annotations(annotation_file)
    
    # Set up the CSV file and writer
    fieldnames = ['filename', 'gt_num_faces', 'response_num_faces', 'raw_response']
    
    # Check if the file exists to determine if we need to write a header
    file_exists = os.path.isfile(csv_filename)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    
    # Open the file in append mode so we can add rows incrementally
    csv_file = open(csv_filename, 'a', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write header only if file is new
    if not file_exists:
        writer.writeheader()
    
    # If file exists, determine which images we've already processed
    processed_images = set()
    if file_exists:
        with open(csv_filename, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_images.add(row['filename'])
    
       
    try:
        # Process each image
        for filename, gt_num_faces in tqdm(annotations.items()):
            # Skip if this image was already processed
            if filename in processed_images:
                continue
            
            # Load image
            img_path = os.path.join(dataset_dir, filename)
            img = cv2.imread(img_path)
            
            # Skip if image loading failed
            if img is None:
                print(f"Warning: Failed to load image {img_path}, skipping")
                continue

            # Convert from BGR (OpenCV format) to RGB (for PIL)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_img = Image.fromarray(img_rgb)
            max_size = 5 * 1024 * 1024  # 5MB in bytes
            current_width, current_height = pil_img.size
            scale_factor = 1.0

            while True:
                # Save to bytes buffer in PNG format
                buffer = io.BytesIO()
                pil_img.save(buffer, format="PNG")
                buffer.seek(0)
    
                # Check size
                if len(buffer.getvalue()) <= max_size:
                    break
    
                # Resize by 50%
                scale_factor *= 0.5
                new_width = int(current_width * scale_factor)
                new_height = int(current_height * scale_factor)
                pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
    
                print(f"Resizing image to {new_width}x{new_height} (scale: {scale_factor:.2f})")
    
                # Safety check to prevent infinite loop with images that can't be compressed enough
                if new_width < 200 or new_height < 200:
                    print(f"Warning: Cannot reduce image {img_path} below 5MB even at very small size")
                    break
            buffer.seek(0)
    
            # Encode to base64
            encoded_file = base64.b64encode(buffer.getvalue()).decode("utf-8")
            base64_url = f"data:image/png;base64,{encoded_file}"
            
            # Prepare prompt
            prompt = [
                {"role": "system", "content": "You are an AI assistant that specializes in analyzing images. Please provide accurate and concise information."},
                {"role": "user", "content": [
                    {"type": "text", "text": "How many visible human faces are in this image? Please respond with just a number. Any response other than an integer will be considered an error."},
                    {"type": "image_url", "image_url": {"url": base64_url}}
                ]}
            ]
            
            try:
                # Make API call
                response = completion(model=model_name, messages=prompt, temperature=0.0, max_tokens=10)
                response_text = response.choices[0].message.content
                parsed_response = parse_face_count_response(response_text)
                try:
                    clean_response = response_text.strip().replace('\n', ' ').replace('\r', ' ')
                except:
                    clean_response = ""

                # Create result dictionary
                result = {
                    'filename': filename,
                    'gt_num_faces': gt_num_faces,
                    'response_num_faces': parsed_response,
                    'raw_response': clean_response
                }
                
                
                # Immediately write this result to the CSV file
                writer.writerow(result)
                # Flush to make sure it's written to disk
                csv_file.flush()
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    finally:
        # Close the CSV file
        csv_file.close()
    
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

    # Load XAI API key if file exists
    if os.path.exists("keys/xai_key.txt"):
        with open("keys/xai_key.txt", "r") as f:
            os.environ["XAI_API_KEY"] = f.read().strip()

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLMs on face counting task')
    parser.add_argument('--model', type=str, required=True, 
                        help='Model identifier (e.g., "openai/gpt-4o")')
    parser.add_argument('--dataset_dir', type=str, default='./WIDER_val/images',
                        help='Directory containing WiderFace images (default: ./WIDER_val/images)')
    parser.add_argument('--annotation_file', type=str, default='./WIDER_val/wider_face_val_bbx_gt.txt',
                        help='Path to WiderFace annotation file (default: ./WIDER_val/wider_face_val_bbx_gt.txt)')
    
    args = parser.parse_args()
    
    # Load API keys
    load_api_keys()
    
    # Run evaluation
    evaluate_face_counter(args.model, args.dataset_dir, args.annotation_file)

if __name__ == "__main__":
    main()