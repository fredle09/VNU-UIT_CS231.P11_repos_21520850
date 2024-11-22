import numpy as np
from scipy.ndimage import convolve
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import argparse
import logging


PATH_FOLDER = "images"
NAME_FOLDERS = ["train", "test"]


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise Exception(f"Error: Unable to read image at {path}")
    return img


def sobel_filters(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    s_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    s_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    i_x = convolve(img, s_x)
    i_y = convolve(img, s_y)

    g = np.hypot(i_x, i_y)
    g = g / g.max() * 255

    theta = np.arctan2(i_y, i_x)

    return i_x, i_y, g, theta


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def min_max_normalize(vector: np.ndarray) -> np.ndarray:
    min_val = np.min(vector)
    max_val = np.max(vector)
    if max_val - min_val == 0:
        return vector
    return (vector - min_val) / (max_val - min_val)


def parse_image_to_vector(img: np.ndarray) -> np.ndarray:
    img_ = cv2.resize(img, (256, 256))
    _, _, g, _ = sobel_filters(img_)
    row_sum = np.sum(g, axis=1, dtype=np.float64)
    col_sum = np.sum(g, axis=0, dtype=np.float64)
    vector = np.hstack((row_sum.T, col_sum))
    return min_max_normalize(vector)


def load_all_images_path_from_folder(folder_path: str) -> list:
    images_path = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            images_path.append(img_path)
    return images_path


def parse_images_to_vectors_and_label(images_path: list) -> tuple[list, list]:
    images = []
    labels = []
    for img_path in images_path:
        img = load_image(img_path)
        img_vector = parse_image_to_vector(img)
        images.append(img_vector)
        labels.append(img_path.split("\\")[-1].split("-")[0])
    return images, labels


def display_images_side_by_side(img_src, img_rst, title1, title2):
    plt.figure(figsize=(20, 20))
    # show img src
    plt.subplot(1, 2, 1)
    plt.title(title1)
    img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    plt.imshow(img_src, interpolation="bicubic")
    # show img result
    plt.subplot(1, 2, 2)
    plt.title(title2)
    img_rst = cv2.cvtColor(img_rst, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rst, interpolation="bicubic")
    plt.show()


def store_features_vectors_and_labels(
    images: list, labels: list[str], suffix: str
) -> None:
    np.save(f"images_{suffix}.npy", images)
    np.save(f"labels_{suffix}.npy", labels)


def load_features_vectors_and_labels(suffix: str) -> tuple:
    if not os.path.exists(f"images_{suffix}.npy") or not os.path.exists(
        f"labels_{suffix}.npy"
    ):
        raise Exception(f"Error: Unable to load images and labels from {suffix} folder")
        # Please run the mode 'load_image_and_store_vector' first

    images = np.load(f"images_{suffix}.npy")
    labels = np.load(f"labels_{suffix}.npy")
    return images, labels


def one_hot_encoding_labels(labels: list[str]) -> np.ndarray:
    unique_labels = np.unique(labels)
    encoded_labels = np.zeros((len(labels), len(unique_labels)))
    for i, label in enumerate(labels):
        encoded_labels[i, np.where(unique_labels == label)] = 1
    return encoded_labels


def order_encoding_labels(labels: list[str]) -> np.ndarray:
    unique_labels = np.unique(labels)
    encoded_labels = np.zeros(len(labels))
    for i, label in enumerate(labels):
        encoded_labels[i] = np.where(unique_labels == label)[0][0]
    return encoded_labels


def reverse_order_encoding_labels(
    encoded_labels: np.ndarray, unique_labels: list[str]
) -> list:
    labels = []
    for i in range(len(encoded_labels)):
        labels.append(unique_labels[int(encoded_labels[i])])
    return labels


def train_knn_classifier(x_train, y_train, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    return knn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["1", "2", "all"],
        default="all",
        help="Stage to run (1, 2, or all)",
    )
    parser.add_argument(
        "-k", type=int, default=3, help="Number of neighbors for kNN classifier"
    )
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))

    stage = args.stage
    k = args.k

    if stage == "1" or stage == "all":
        # Load images
        for folder_name in NAME_FOLDERS:
            folder_path = os.path.join(PATH_FOLDER, folder_name)
            images_path = load_all_images_path_from_folder(folder_path)
            images, labels = parse_images_to_vectors_and_label(images_path)
            store_features_vectors_and_labels(images, labels, folder_name)
            logging.info(
                f"Images and labels from {folder_name} folder parsed successfully"
            )

    if stage == "2" or stage == "all":
        x_train, y_train = load_features_vectors_and_labels("train")
        y_train_encoded = order_encoding_labels(y_train)

        knn = train_knn_classifier(x_train, y_train_encoded, k)
        logging.info(f"k-NN classifier trained successfully with k={k}")

        x_test, y_test = load_features_vectors_and_labels("test")
        y_test_encoded = order_encoding_labels(y_test)
        predictions = knn.predict(x_test)
        logging.debug("Predictions: %s", predictions)
        logging.debug("Actual labels: %s", y_test_encoded)

        cm = confusion_matrix(y_test_encoded, predictions)
        logging.info("Confusion Matrix:\n%s", cm)

        accuracy = np.trace(cm) / np.sum(cm)
        logging.info("Accuracy: %.2f%%", accuracy * 100)


if __name__ == "__main__":
    main()
