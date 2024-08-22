import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# Configuration
model_path = 'fasterrcnn_model.pth'  # Path to the saved model
image_folder = '/Users/amandanassar/Desktop/V60 NOK AI/seg_pred'  # Folder with images to predict
output_folder = '/Users/amandanassar/Desktop/V60 NOK AI/predictions'  # Folder to save predictions

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = fasterrcnn_resnet50_fpn(pretrained=False)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Define a function to process a single image
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        predictions = model(image_tensor)

    return predictions[0]  # Extract the first (and only) image prediction

# Process each image in the folder
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    predictions = predict_image(img_path)

    # Process the predictions (e.g., draw bounding boxes)
    image = Image.open(img_path).convert("RGB")
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Convert predictions to numpy arrays for easy processing
    scores = predictions['scores'].cpu().numpy()
    boxes = predictions['boxes'].cpu().numpy()

    for i in range(len(scores)):
        if scores[i] > 0.5:  # Filter out low-confidence predictions
            box = boxes[i]
            x_min, y_min, x_max, y_max = box
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                              edgecolor='red', facecolor='none', linewidth=2))

    plt.axis('off')
    plt.savefig(os.path.join(output_folder, img_name))
    plt.close()

print("Predictions saved in:", output_folder)
