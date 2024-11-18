import os

def calculate_precision_recall_fscore(retrieved_names, relevant, top_k=5):
    # Limit retrieved to top_k
    retrieved = [os.path.basename(image_path) for image_path in retrieved_names]
    retrieved = set(retrieved[:top_k])
    relevant = set(relevant)

    # Intersection of retrieved and relevant images
    true_positives = len(retrieved & relevant)
    precision = true_positives / len(retrieved) if retrieved else 0
    recall = true_positives / len(relevant) if relevant else 0
    f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f_score
