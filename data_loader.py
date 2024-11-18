import os
import numpy as np
import pickle
import json
from pycocotools.coco import COCO
from config import CAPTIONS_PATH, IMAGE_DIR, INSTANCES_PATH

def load_and_combine_coco_data(instances_path=INSTANCES_PATH, captions_path=CAPTIONS_PATH):
    # Load category and image information from instances_val2017.json
    with open(instances_path, 'r') as f:
        instances_data = json.load(f)
    
    # Load caption data from captions_val2017.json
    with open(captions_path, 'r') as f:
        captions_data = json.load(f)
    
    # Create a mapping from image ID to category IDs (category_id)
    image_id_to_file_name = {
        str(image['id']): image['file_name']
        for image in instances_data['images']
    }

    image_to_categories = {}
    for annotation in instances_data['annotations']:
        image_id = str(annotation['image_id'])
        category_id = annotation['category_id']
        if image_id not in image_to_categories:
            image_to_categories[image_id] = set()
        image_to_categories[image_id].add(category_id)
    
    image_to_categories = {
        image_id: list(categories)
        for image_id, categories in image_to_categories.items()
    }

    image_to_captions = {}
    for annotation in captions_data['annotations']:
        image_id = str(annotation['image_id'])
        caption = annotation['caption']
        if image_id not in image_to_captions:
            image_to_captions[image_id] = annotation['caption']
    
    combined_data = []
    for image_id, file_name in image_id_to_file_name.items():
        combined_data.append({
            'image_name': file_name,
            'categories': image_to_categories.get(image_id, []),
            'captions': image_to_captions.get(image_id, [])
        })
    
    return combined_data


def save_text_features(text_features, vectorizer, features_dir="features"):
    os.makedirs(features_dir, exist_ok=True)

    np.save(os.path.join(features_dir, "text_features.npy"), text_features)

    with open(os.path.join(features_dir, "vectorizer.pkl"), 'wb') as f:
        pickle.dump(vectorizer, f)

    print("Features saved successfully!")


def save_image_features(image_features, features_dir="features"):
    os.makedirs(features_dir, exist_ok=True)

    np.save(os.path.join(features_dir, "image_features.npy"), image_features)

    print("Features saved successfully!")


def gather_image_paths(combined_data):
    image_paths = []

    for entry in combined_data:
        img_path = os.path.join(IMAGE_DIR, entry['image_name'])  
        image_paths.appenfd(img_path)

    return image_paths


def load_text_features(features_dir="features"):
    text_features_path = os.path.join(features_dir, "text_features.npy")
    vectorizer_path = os.path.join(features_dir, "vectorizer.pkl")

    if not os.path.exists(text_features_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Text features or vectorizer not found in the specified directory.")

    text_features = np.load(text_features_path)

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    print("Text features and vectorizer loaded successfully!")
    return text_features, vectorizer


def load_image_features(features_dir="features"):
    image_features_path = os.path.join(features_dir, "image_features.npy")

    if not os.path.exists(image_features_path):
        raise FileNotFoundError("Image features not found in the specified directory.")

    image_features = np.load(image_features_path)

    print("Image features loaded successfully!")
    return image_features