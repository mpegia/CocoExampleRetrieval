import os
from config import TOP_K, IMAGE_DIR
import data_loader
import retrieval
import helper
from descriptor import extract_text_features, extract_image_features, get_image_paths
from evaluation import calculate_precision_recall_fscore

if __name__ == '__main__':
    combined_data = data_loader.load_and_combine_coco_data()

    # Extract and save text features
    # vectorizer, text_features = extract_text_features(combined_data)
    text_features, vectorizer = data_loader.load_text_features()
    # data_loader.save_text_features(text_features, vectorizer, features_dir="features")
    
    # Extract and save image features
    # image_paths, image_features = extract_image_features(combined_data)
    image_features = data_loader.load_image_features()
    image_paths = get_image_paths(combined_data)
    # data_loader.save_image_features(image_features, features_dir="features")

    # Queries
    image_query = '000000231237.jpg'
    text_query = "A vase with various flowers in it on a display case."
    multimodal_query = [image_query, text_query]

    # Evaluate retrieval methods
    retrieved_images1 = retrieval.evaluate_image_retrieval(image_query, combined_data, image_paths, image_features, TOP_K)
    retrieved_images2 = retrieval.evaluate_text_to_image_retrieval(text_query, combined_data, image_paths, text_features, vectorizer, TOP_K)
    retrieved_images3 = retrieval.evaluate_text_image_to_image_retrieval(multimodal_query, combined_data, image_paths, text_features, vectorizer, image_features, TOP_K)

    retrieved_results = [retrieved_images1, retrieved_images2, retrieved_images3]
    method_names = ["Image-to-Image", "Text-to-Image", "Multimodal Query"]
    helper.plot_retrieval_results(image_query, retrieved_results, top_k=5, method_names=method_names)