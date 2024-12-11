# -*- coding: utf-8 -*-
"""
VNU-UIT_CS231.P11
21520850 - Lê Trung Hiếu

-- Kết quả
Logistic Regression - none              - Accuracy:     1.0
K Neatest Neighbor  - none       - k: 1 - Accuracy:     1.0
K Neatest Neighbor  - none       - k: 3 - Accuracy:     1.0
K Neatest Neighbor  - none       - k: 5 - Accuracy:     1.0
K Neatest Neighbor  - none       - k: 7 - Accuracy:     1.0
K Neatest Neighbor  - none       - k: 9 - Accuracy:     0.875

Logistic Regression - normalize         - Accuracy:     0.75
K Neatest Neighbor  - normalize  - k: 1 - Accuracy:     1.0
K Neatest Neighbor  - normalize  - k: 3 - Accuracy:     1.0
K Neatest Neighbor  - normalize  - k: 5 - Accuracy:     1.0
K Neatest Neighbor  - normalize  - k: 7 - Accuracy:     1.0
K Neatest Neighbor  - normalize  - k: 9 - Accuracy:     0.875

Logistic Regression - min-max           - Accuracy:     1.0
K Neatest Neighbor  - min-max    - k: 1 - Accuracy:     0.875
K Neatest Neighbor  - min-max    - k: 3 - Accuracy:     0.875
K Neatest Neighbor  - min-max    - k: 5 - Accuracy:     0.875
K Neatest Neighbor  - min-max    - k: 7 - Accuracy:     0.875
K Neatest Neighbor  - min-max    - k: 9 - Accuracy:     1.0

-- Nhận xét:
Logistic Regression cho kết quả tốt nhất với cả ba phương pháp chuẩn hóa và accuracy = 1.0
Còn lại với ít dữ liệu như vậy thì KNN cũng cho kết quả tốt với k = 1, 3, 5, 7 và 9
"""

from enum import Enum
import numpy as np
import cv2
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DEFAULT_SIZE = (128, 128)
PATH_FOLDER = "datasets"
LABELS = ["non-pedestrian", "pedestrian"]


class NormalizeMethod(Enum):
    NONE = "none"
    NORMALIZE = "normalize"
    MIN_MAX = "min-max"


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Error: Unable to read image at {path}")
    return img


def resize_image(img: np.ndarray, size: tuple = DEFAULT_SIZE) -> np.ndarray:
    return cv2.resize(img, size)


def extract_hog_features(img, win_size=(64, 128), cell_size=(8, 8), block_size=(16, 16), nbins=9):
    hog = cv2.HOGDescriptor(win_size, block_size, cell_size, cell_size, nbins)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return hog.compute(img).flatten()


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


def parse_image_to_vector(img: np.ndarray, normalize_method: NormalizeMethod) -> np.ndarray:
    img = resize_image(img)
    vector = extract_hog_features(img)

    if normalize_method == NormalizeMethod.NONE:
        return vector
    elif normalize_method == NormalizeMethod.NORMALIZE:
        return normalize_vector(vector)
    elif normalize_method == NormalizeMethod.MIN_MAX:
        return min_max_normalize(vector)
    raise NotImplementedError("Error: Normalize method not implemented")


def load_all_images_path_from_folder(name_folder: str) -> list:
    images_path = []
    folder = f"{PATH_FOLDER}/{name_folder}"
    for path in os.listdir(folder):
        images_path.append((f"{folder}/{path}", name_folder))
    return images_path


def parse_images_to_vectors_and_label(
    images_path: list, normalize_method: NormalizeMethod
) -> tuple:
    vectors = []
    labels = []
    for path, label in images_path:
        img = load_image(path)
        vector = parse_image_to_vector(img, normalize_method)
        vectors.append(vector)
        labels.append(label)
    return vectors, labels


def train_logistic_regression(X_train, y_train, max_iter=100):
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train, n_neighbors=3):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def plot_accuracy(results):
    for method, accuracies in results.items():
        plt.plot(accuracies['k_values'], accuracies['scores'], marker='o', label=method)

    plt.xlabel('k (for kNN) / Logistic Regression')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.legend()
    plt.show()


def main():
    images_path = [*load_all_images_path_from_folder("non-pedestrian"),
                   *load_all_images_path_from_folder("pedestrian")]
    for normalize_method in NormalizeMethod:
        X, y = parse_images_to_vectors_and_label(images_path, normalize_method)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = train_logistic_regression(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        print(f"Logistic Regression - {normalize_method.value}\t        - Accuracy: \t{accuracy:2}")

        for i in [1, 3, 5, 7, 9]:
            model = train_knn(X_train, y_train, i)
            accuracy = evaluate_model(model, X_test, y_test)
            print(f"K Neatest Neighbor  - {normalize_method.value}\t - k: {i} - Accuracy: \t{accuracy:2}")


if __name__ == "__main__":
    main()