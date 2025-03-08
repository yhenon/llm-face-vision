import cv2
import base64
import cv2
import numpy as np
import io
from PIL import Image

def prepare_images_for_llm(img1, img2):
    """
    Concatenate two images horizontally, encode as base64, and return properly formatted content
    for LLM API consumption.
    
    Args:
        img1 (numpy.ndarray): First image as a cv2/numpy array
        img2 (numpy.ndarray): Second image as a cv2/numpy array
        
    Returns:
        list: Formatted content list ready for LLM API
    """
    
    # Concatenate the images horizontally with the gap
    combined_img = np.hstack((img1, img2))
    
    # Convert from BGR (OpenCV format) to RGB (for PIL)
    combined_img_rgb = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_img = Image.fromarray(combined_img_rgb)
    
    # Save to bytes buffer in PNG format
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    
    # Encode to base64
    encoded_file = base64.b64encode(buffer.getvalue()).decode("utf-8")
    base64_url = f"data:image/png;base64,{encoded_file}"
    
    # Format for API
    image_content = [
        {"type": "text", "text": "The image contains a face on the left and a face on the right. Are these two faces of the same person? Respond only with YES or NO. Any answer other than YES or NO will be considered a failure."},
        {
            "type": "image_url",
            "image_url": {"url": base64_url}
        }
    ]

    prompt = [{"role": "user", "content": image_content}]
    
    return prompt

def parse_response(response):
    response = response.strip()
    if response.lower() == 'yes':
        return 1
    elif response.lower() == 'no':
        return 0
    else:
        return -1