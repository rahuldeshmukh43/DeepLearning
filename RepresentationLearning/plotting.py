
import numpy as np

import IPython.display as display
from PIL import Image
from io import BytesIO

import umap
import umap.plot
import matplotlib.pyplot as plt

def plot_umap(embeddings, labels=None, title=""):
    """
    Return:
        bug (np array) [h, w, 3] rgb array
    """
    embeddings_umap = umap.UMAP().fit_transform(embeddings)
    # convert fig to img
    fig, ax = plt.subplots(figsize=(5, 5))
    if isinstance(labels, np.ndarray):
        scatter = ax.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], c=labels, cmap='Spectral', s=5)
    else:
        scatter = ax.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], s=5)
    if title != "": ax.set_title(title)
    # Save the plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    # Load buffer as an image
    img = Image.open(buf)
    return img

def plot_image_grid(images, grid_width=4, figsize=(15, 10)):
    """
    Plots a list of PIL Image objects in a grid with a fixed width.

    Parameters:
    - images: List of PIL Image objects.
    - grid_width: Number of images per row (default is 4).
    - figsize: Size of the entire figure (default is (15, 10)).

    Returns:
    - None
    """
    # Calculate the number of rows needed
    grid_height = int(np.ceil(len(images) / grid_width))

    # Create a figure with subplots
    fig, axes = plt.subplots(grid_height, grid_width, figsize=figsize)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot each image
    for i, img in enumerate(images):
        # Turn off axis for the current subplot
        axes[i].axis('off')
        # Display the image in the current subplot
        axes[i].imshow(img)

    # Turn off axis for any remaining subplots if images < grid size
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def create_gif_from_rgb_list(images, duration=500):
    """
    Creates a GIF from a list of PIL images and returns it as a byte buffer.

    Parameters:
    - images: List of PIL Image objects.
    - duration: Duration for each frame in milliseconds.

    Returns:
    - A BytesIO object containing the GIF data.
    """
    gif_buffer = BytesIO()
    images[0].save(
        gif_buffer,
        format='GIF',
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    
    # Ensure the buffer's pointer is at the start
    gif_buffer.seek(0)
    return gif_buffer

def display_gif(gif_buffer):
    """
    Displays a GIF stored in a BytesIO buffer inline in a Jupyter Notebook.

    Parameters:
    - gif_buffer: BytesIO object containing the GIF data.
    """
    display.display(display.Image(data=gif_buffer.getvalue()))
    return