import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

DIR = "datasets"  # Replace with your folder containing images


def load_all_images_path_from_folder(folder_path: str = DIR) -> list:
    images_path = []
    for path in os.listdir(folder_path):
        images_path.append(f"{folder_path}/{path}")
    return images_path


def K_Means(img, n_clusters=6):
    nRow, nCol, nChl = img.shape
    g = img.reshape(nRow * nCol, nChl)
    k_means = KMeans(n_clusters=n_clusters, random_state=0).fit(g)
    clustered = k_means.cluster_centers_[k_means.labels_]
    img_res = clustered.reshape(nRow, nCol, nChl).astype(np.uint8)
    return img_res


def K_Means_2(img, n_clusters=6):
    nRow, nCol, nChl = img.shape
    # Represent pixel as (r, g, b, x, y)
    g = []
    for y in range(nRow):
        for x in range(nCol):
            g.append([img[y, x, 0], img[y, x, 1], img[y, x, 2], x, y])
    g = np.array(g)

    k_means = KMeans(n_clusters=n_clusters, random_state=0).fit(g)
    clustered = k_means.cluster_centers_[k_means.labels_]

    # Map back to (r, g, b)
    img_res = np.zeros((nRow, nCol, nChl), dtype=np.uint8)
    index = 0
    for y in range(nRow):
        for x in range(nCol):
            img_res[y, x] = clustered[index, :3]
            index += 1
    return img_res


def globalThresholding(img, thres=127):
    img_rst = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img_rst[i][j] < thres:
                img_rst[i][j] = 255
            else:
                img_rst[i][j] = 0
    return img_rst


def adaptiveThreshold(f, nRow, nCol):
    g = f.copy()
    r = int(f.shape[0] / nRow)
    c = int(f.shape[1] / nCol)
    for i in range(int(nRow)):
        for j in range(int(nCol)):
            x = f[i * r : (i + 1) * r, j * c : (j + 1) * c]
            t = int(np.mean(x))
            g[i * r : (i + 1) * r, j * c : (j + 1) * c] = globalThresholding(x, t)
    return g


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


TESTING_CONFIG = {
    globalThresholding: [
        "particles.bmp",
        "Phandoan01.jpg",
        "wdg2.jpg",
        "Rice.png",
    ],
    adaptiveThreshold: ["wdg3.jpg"],
    K_Means: ["vegetables.jpg", "hand.jpg", "thuoc.jpg"],
    K_Means_2: ["vegetables.jpg", "thuoc.jpg"],
}


def main():
    list_images = []
    titles = []
    
    # Collect all images and their processed versions
    for func, filenames in TESTING_CONFIG.items():
        for filename in filenames:
            img_path = os.path.join(DIR, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image {img_path}")
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            list_images.append(img_rgb)  # Add original image
            titles.append(f"{filename} - Original")

            # Process image with the respective function
            if func in [globalThresholding, adaptiveThreshold]:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if func == globalThresholding:
                    processed_img = func(img_gray, thres=127)
                else:
                    processed_img = func(img_gray, nRow=4, nCol=4)
            else:
                processed_img = func(img_rgb, n_clusters=6)

            list_images.append(processed_img)  # Add processed image
            titles.append(f"{func.__name__} Processed")

    # Calculate grid size
    n_images = len(list_images)
    cols = 4
    rows = (n_images + 1) // cols  # Each row has 2 images (original + processed)

    # Visualize all images in a grid
    visualize_images_grid(list_images, titles, rows, cols)


if __name__ == "__main__":
    main()
