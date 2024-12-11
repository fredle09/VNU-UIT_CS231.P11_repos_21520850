import cv2
import numpy as np
import matplotlib.pyplot as plt

DIR = "datasets"  # Replace with your folder containing images


def apply_grabcut(img, rect=None):
    # Create an initial mask, where 0 represents background, and 2 represents probable background
    mask = np.zeros(img.shape[:2], np.uint8)

    # Create background and foreground models (these will be updated by GrabCut)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Apply GrabCut algorithm
    # If no rectangle is provided, grabcut will try to segment based on the initial mask
    if rect is None:
        # Automatically generate a rectangle that encompasses the whole image
        rect = (5, 5, img.shape[1] - 5, img.shape[0] - 5)

    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask to obtain the foreground (fgd = 1, probable fgd = 0)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    # Segment the image
    segmented_img = img * mask2[:, :, np.newaxis]

    return segmented_img, mask2


def show_images(
    original_img, segmented_img, title1="Original Image", title2="Segmented Image"
):
    """
    Show original and segmented images side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(title1)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(title2)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


# def load_all_images_path_from_folder(name_folder: str = DIR) -> list:
#     images_path = []
#     for path in os.listdir(name_folder):
#         images_path.append(f"{name_folder}/{path}")
#     return images_path


def visualize_images_grid(images, titles, rows, cols, wspace=0.5, hspace=0.5):
    """
    Display images in a grid with titles and customizable gaps between them.
    
    Parameters:
    - wspace: Horizontal space between images (default is 0.2).
    - hspace: Vertical space between images (default is 0.3).
    """
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    # Display images and set titles
    for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
        ax.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
        ax.set_title(title)
        ax.axis("off")

    # Hide any remaining axes if there are more axes than images
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust space between the subplots
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    plt.tight_layout()
    plt.show()


TEST_CONFIG = {
    # f"{DIR}/airplane.jpg": [100, 50, 300, 400],
    # f"{DIR}/bongda.jpg": [100, 50, 300, 400],
    f"{DIR}/camourflage_00012.jpg": [100, 0, 600, 400],
    f"{DIR}/camourflage_00097.jpg": [100, 50, 235, 200],
    f"{DIR}/camourflage_00129.jpg": [100, 30, 350, 200],
    f"{DIR}/camourflage_00166.jpg": [20, 80, 500, 280],
    f"{DIR}/camourflage_00197.jpg": [50, 75, 700, 470],
    # f"{DIR}/cell.tif": [100, 50, 300, 400],
    f"{DIR}/dogcat.jpg": [0, 0, 125, 160],
    # f"{DIR}/eight.tif": [100, 50, 300, 400],
    # f"{DIR}/hand.jpg": [100, 50, 300, 400],
    f"{DIR}/Lionel-Messi.jpg": [230, 10, 320, 400],
    # f"{DIR}/particles.bmp": [100, 50, 300, 400],
    # f"{DIR}/Phandoan01.jpg": [100, 50, 300, 400],
    # f"{DIR}/rice.png": [100, 50, 300, 400],
    # f"{DIR}/son1.jpg": [100, 50, 300, 400],
    # f"{DIR}/thuoc.jpg": [100, 50, 300, 400],
    # f"{DIR}/vegetables.jpg": [100, 50, 300, 400],
    # f"{DIR}/wdg2.jpg": [100, 50, 300, 400],
    # f"{DIR}/wdg3.jpg": [100, 50, 300, 400],
}


def main():
    images = []
    titles = []
    for img_filename in TEST_CONFIG:
        img = cv2.imread(img_filename)
        if img is None:
            print(f"Failed to load image {img}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        rect = TEST_CONFIG[img_filename]
        if rect is None:
            print(f"Failed to load rectangle for image {img}")
            continue
        segmented_img, mask = apply_grabcut(img, rect)
        if segmented_img is None:
            continue
        
        images.append(img)
        images.append(segmented_img)
        titles.append(img_filename)
        titles.append("Segmented")
    
    cols = 4
    rows = len(images) // cols + bool(len(images) % cols)
    visualize_images_grid(images, titles, rows, cols)


if __name__ == "__main__":
    main()
