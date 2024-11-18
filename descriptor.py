import numpy as np
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
from config import TOP_K, IMAGE_DIR

# Load the pre-trained VGG19 model
vgg19 = models.vgg19(pretrained=True)
vgg19.eval()  # Set the model to evaluation mode

# Define the transformation pipeline (resizing, normalization, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (required by VGG19)
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for VGG19
])

def get_image_paths(combined_data):
    image_paths = []
    for entry in combined_data:
        img_path = os.path.join(IMAGE_DIR, entry['image_name'])  # Get the image path
        image_paths.append(img_path)
    return image_paths


def extract_image_feature(query):
    img_path = os.path.join(IMAGE_DIR, query)  
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)
        
    # Check if GPU is available and move the tensor to GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    vgg19.to(device)
        
    # Extract features with no gradient computation
    with torch.no_grad():
        image_feature = vgg19(image)
        
    # Flatten the features and append to the list
    image_feature = image_feature.flatten().cpu().numpy()  # Move back to CPU if needed
    
    image_feature = np.array(image_feature)
    
    return img_path, image_feature


def extract_image_features(combined_data):
    image_features = []
    image_paths = []

    for entry in combined_data:
        img_path = os.path.join(IMAGE_DIR, entry['image_name'])  
        image_paths.append(img_path)

        # Open the image and apply the transformation
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)
        
        # Check if GPU is available and move the tensor to GPU if possible
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image = image.to(device)
        vgg19.to(device)
        
        # Extract features with no gradient computation
        with torch.no_grad():
            features = vgg19(image)
        
        features = features.flatten().cpu().numpy()  # Move back to CPU if needed
        image_features.append(features)
    
    image_features = np.array(image_features)
    
    return image_paths, image_features


def extract_text_features(combined_data):
    captions = []
    for item in combined_data:
        if item['captions']:  
            captions.append(item['captions'])

    vectorizer = TfidfVectorizer(max_features=300)
    text_features = vectorizer.fit_transform(captions).toarray()
    return vectorizer, text_features
