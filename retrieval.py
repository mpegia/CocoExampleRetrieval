from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import helper
import evaluation
import descriptor


def get_relevant_images(image_query, query_text, combined_data):
    if image_query:
        query_categories = helper.get_categories(image_query, 'image', combined_data)
    elif query_text:
        query_categories = helper.get_categories(query_text, 'text', combined_data)
    else:
        return []

    relevant_images = []
    for entry in combined_data:
        image_name = entry['image_name']
        image_categories = entry.get('categories', [])

        if any(category in query_categories for category in image_categories):
            relevant_images.append(image_name)
    
    return relevant_images

def image_to_image_retrieval(image_query, image_paths, image_features, top_k):
    query_image_path, query_image_feature = descriptor.extract_image_feature(image_query)  
    similarities = cosine_similarity([query_image_feature], image_features) 
    sorted_indices = np.argsort(similarities[0])[::-1]  
    return [image_paths[i] for i in sorted_indices[:top_k]]


def evaluate_image_retrieval(image_query, combined_data, image_paths, image_features, top_k):
    retrieved_images = image_to_image_retrieval(image_query, image_paths, image_features, top_k)
    relevant_images = get_relevant_images(image_query, None, combined_data)
    precision, recall, f_score = evaluation.calculate_precision_recall_fscore(retrieved_images, relevant_images, top_k)

    print(f"Evaluation Results for Image to Image Retrieval (Top-{top_k}):")
    print(f"  Precision@{top_k}: {precision:.2f}")
    print(f"  Recall@{top_k}: {recall:.2f}")
    print(f"  F-Score: {f_score:.2f}")

    return retrieved_images

def text_to_image_retrieval(query_text, image_paths, text_features, vectorizer, top_k):
    query_text_vectorized = vectorizer.transform([query_text])  
    similarities = cosine_similarity(query_text_vectorized, text_features) 
    sorted_indices = np.argsort(similarities[0])[::-1]  
    return [image_paths[i] for i in sorted_indices[:top_k]]


def evaluate_text_to_image_retrieval(text_query, combined_data, image_paths, text_features, vectorizer, top_k):
    retrieved_images = text_to_image_retrieval(text_query, image_paths, text_features, vectorizer, top_k)
    relevant_images = get_relevant_images(None, text_query, combined_data)
    precision, recall, f_score = evaluation.calculate_precision_recall_fscore(retrieved_images, relevant_images, top_k)

    print(f"Evaluation Results for Text to Image Retrieval (Top-{top_k}):")
    print(f"  Precision@{top_k}: {precision:.2f}")
    print(f"  Recall@{top_k}: {recall:.2f}")
    print(f"  F-Score: {f_score:.2f}")

    return retrieved_images


def text_image_to_image_retrieval(query_image_path, query_text, image_paths, text_features, vectorizer, image_features, top_k):
    query_image_feature = descriptor.extract_image_feature(query_image_path)  
    query_text_vectorized = vectorizer.transform([query_text])  
    
    combined_features = np.concatenate((query_image_feature[1], query_text_vectorized.toarray().flatten()))
    
    combined_image_features = np.concatenate((image_features, text_features), axis=1)
    similarities = cosine_similarity([combined_features], combined_image_features)
    sorted_indices = np.argsort(similarities[0])[::-1]
    return [image_paths[i] for i in sorted_indices[:top_k]]


def evaluate_text_image_to_image_retrieval(multimodal_query, combined_data, image_paths, text_features, vectorizer, image_features, top_k):
    [image_query, text_query] = multimodal_query
    retrieved_images = text_image_to_image_retrieval(image_query, text_query, image_paths, text_features, vectorizer, image_features, top_k)
    relevant_images = get_relevant_images(image_query, text_query, combined_data)
    precision, recall, f_score = evaluation.calculate_precision_recall_fscore(retrieved_images, relevant_images, top_k)

    print(f"Evaluation Results for (Text, Image) to Image Retrieval (Top-{top_k}):")
    print(f"  Precision@{top_k}: {precision:.2f}")
    print(f"  Recall@{top_k}: {recall:.2f}")
    print(f"  F-Score: {f_score:.2f}")

    return retrieved_images