import os
import numpy as np
import csv
import torch
from insightface.model_zoo import get_model
import cv2


def extract_features(model, image_path):
    """Extract feature vector from a pre-aligned face image"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None
    
    # Convert BGR to RGB (InsightFace models expect RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to float32
 
    img = img.astype(np.float32)

    embedding = model.forward(np.expand_dims(img, axis=0).transpose(0, 3, 1, 2))[0,:]
    
    return embedding

def compute_similarity(feat1, feat2):
    """Compute cosine similarity between two feature vectors"""
    if feat1 is None or feat2 is None:
        return -1  # Return -1 for failed extractions
    
    # Normalize the vectors
    feat1 = feat1 / np.linalg.norm(feat1)
    feat2 = feat2 / np.linalg.norm(feat2)
    
    # Compute cosine similarity
    similarity = np.dot(feat1, feat2)
    return similarity

# Process the pairs file and compute similarities
def process_pairs(pairs_file, output_csv):
    results = []
    # Initialize the ArcFace model directly (without detection)
    model_name = './model/arcface_r100_v1.onnx'  # You can choose different models like 'arcface_r50_v1', etc.
    model = get_model(model_name, root='./model')
    model.prepare(ctx_id=0)  # Use GPU 0

    with open(pairs_file, 'r') as f:
        pairs = f.readlines()
        # Write results to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image1', 'image2', 'is_same', 'similarity'])

        for pair in pairs:
            parts = pair.strip().split()
            is_same = int(parts[0])  # 1 if same person, 0 if different
            img1_path = os.path.join('./val',parts[1])
            img2_path = os.path.join('./val',parts[2])
            
            # Extract features
            feat1 = extract_features(model, img1_path)
            feat2 = extract_features(model, img2_path)
            
            # Compute similarity
            similarity = compute_similarity(feat1, feat2)
            
            # Save result
            result = [img1_path, img2_path, is_same, similarity]
            
            # Print progress
            print(f"Processed pair: {img1_path} - {img2_path}, similarity: {similarity:.8f}")
        
            writer.writerow(result)
        
    print(f"Results saved to {output_csv}")

# Example usage
datasets = ['agedb_30', 'calfw', 'cplfw']
for dataset in datasets:
    pairs_file = f'val/{dataset}_ann.txt'  # Your input file with image pairs
    output_csv = f'arcface/{dataset}_similarities.csv'  # Output file
    process_pairs(pairs_file, output_csv)