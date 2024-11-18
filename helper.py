import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from config import IMAGE_DIR

def get_categories(query, type, combined_data):
    if type == 'image':
        search_keyword = 'image_name'
    else:
        search_keyword = 'captions'
    for entry in combined_data:
        if entry[search_keyword] == query:
            return entry['categories']
        

def plot_retrieval_results(image_query, retrieved_results, top_k, method_names):
    image_query = os.path.join(IMAGE_DIR, image_query)
    num_methods = len(retrieved_results)
    # Dynamic figure size: height proportional to the number of rows
    fig_width = 3 * (top_k + 1)  # Query + retrieved images
    fig_height = 3 * num_methods
    fig, axes = plt.subplots(num_methods, top_k + 1, figsize=(fig_width, fig_height))
    
    # Ensure axes is always a 2D array (even if there's only one method)
    if num_methods == 1:
        axes = [axes]

    fig.suptitle("Retrieval Results", fontsize=16, y=1.02)

    # Load the query image
    if isinstance(image_query, str):  # If it's a file path
        query_img = mpimg.imread(image_query)
    else:  # If it's an array
        query_img = image_query

    for row, (retrieved_images, method_name) in enumerate(zip(retrieved_results, method_names)):
        # Display the query image in the first column of each row
        axes[row][0].imshow(query_img, aspect='auto')
        axes[row][0].axis("off")
        axes[row][0].set_title("Query Image", fontsize=10)

        # Display retrieved images for the current method
        for col in range(top_k):
            ax = axes[row][col + 1]  # Column starts at 1 (after query)
            if col < len(retrieved_images):
                if isinstance(retrieved_images[col], str):  # If it's a file path
                    retrieved_img = mpimg.imread(retrieved_images[col])
                else:  # If it's an array
                    retrieved_img = retrieved_images[col]
                ax.imshow(retrieved_img, aspect='auto')
            else:
                ax.axis("off")  # No image to display
            
            ax.axis("off")
            ax.set_title(f"Rank {col + 1}", fontsize=10)

        # Set method name as the row title
        axes[row][0].set_ylabel(method_name, fontsize=12, rotation=90, labelpad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
    plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Increase spacing between rows and columns
    plt.show()
