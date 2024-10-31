import cv2
import os
import re
from model_predictor import load_model, analyze_image  # Ensure these functions are available
import time
import re  # Ensure the re module is imported

# Define a pattern that matches files like '1 (1).jpg', '1 (2).png', etc.
pattern = re.compile(r'^\d+ \(\d+\)\.(jpg|png)$')

# Set file paths
input_dir = "Dataset/before"      # Folder with images to analyze
output_dir = "Dataset/after"      # Folder to save analyzed images

# Initialize the predictor by loading the model
predictor = load_model("model_final.pth", "cpu")  # Adjust path and device as needed

def process_images(input_dir, output_dir):
    """Process and analyze each image in the input directory, then save to output directory."""
    # Ensure input and output directories exist
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created missing input directory: {input_dir}")
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created missing output directory: {output_dir}")

    # Regex pattern to match files like "1 (1).png", "1 (2).png", etc.
    pattern = re.compile(r"1 \((\d+)\)\.png")

for filename in os.listdir(input_dir):
    print(f"Found file: {filename}")  # Print each file found in the directory
    if pattern.match(filename):
            image_path = os.path.join(input_dir, filename)
            
            try:
                # Analyze the image and save the result
                analyzed_image = analyze_image(predictor, image_path)
                output_path = os.path.join(output_dir, f"analyzed_{filename}")
                cv2.imwrite(output_path, analyzed_image)
                print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

def main():
    """Main function to process images in the folder."""
    print("Starting image analysis...")
    start_time = time.time()
    process_images(input_dir, output_dir)
    end_time = time.time()
    print(f"Completed image analysis in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
