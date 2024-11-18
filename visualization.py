import matplotlib.pyplot as plt
from PIL import Image

def plot_retrieval_results(query, retrieved_images, title="Retrieval Results"):
    plt.figure(figsize=(15, 5))
    plt.suptitle(title)
    
    # Display query image if given
    if isinstance(query, str):  # if query is an image path
        plt.subplot(1, len(retrieved_images) + 1, 1)
        plt.imshow(Image.open(query))
        plt.title("Query Image")
        plt.axis('off')
    
    # Display retrieved images
    for i, img_path in enumerate(retrieved_images):
        plt.subplot(1, len(retrieved_images) + 1, i + 2)
        plt.imshow(Image.open(img_path))
        plt.title(f"Rank {i + 1}")
        plt.axis('off')
    plt.show()
