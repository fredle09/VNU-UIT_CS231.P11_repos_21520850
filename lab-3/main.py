# -*- coding: utf-8 -*-
"""
VNU-UIT_CS231.P11
21520850 - Lê Trung Hiếu
"""

from enum import Enum
import numpy as np
import cv2
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

DEFAULT_SIZE = (256, 256)
PATH_FOLDER = "datasets"
NAME_FOLDERS = ["train", "test"]
LABELS = ["cat", "dog"]


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


def sobel_filters(img: np.ndarray) -> tuple:
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    s_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    s_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    i_x = cv2.filter2D(img, -1, s_x)
    i_y = cv2.filter2D(img, -1, s_y)

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


def extract_color_histogram(img: np.ndarray, bins: int = 16) -> np.ndarray:
    row, col, channel = img.shape[:3]
    size: int = row * col

    feature: list[np.ndarray] = []
    for i in range(channel):
        hist = np.squeeze(cv2.calcHist([img], [i], None, [bins], [0, 256]))
        hist = hist / size
        feature.extend(hist)
    return feature


def parse_image_to_vector(
    img: np.ndarray, normalize_method: NormalizeMethod
) -> np.ndarray:
    img = resize_image(img)
    _, _, g, _ = sobel_filters(img)
    row_sum = np.sum(g, axis=1, dtype=np.float64)
    col_sum = np.sum(g, axis=0, dtype=np.float64)
    color_hist = extract_color_histogram(img)
    vector = np.hstack((row_sum.T, col_sum, color_hist))

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
        images_path.append((f"{folder}/{path}", path.split(".")[0]))
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


def main():
    for normalize_method in NormalizeMethod:
        images_path = load_all_images_path_from_folder("train")
        X_train, y_train = parse_images_to_vectors_and_label(
            images_path, normalize_method
        )
        images_path = load_all_images_path_from_folder("test")
        X_test, y_test = parse_images_to_vectors_and_label(
            images_path, normalize_method
        )

        model = train_logistic_regression(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        print(f"Logistic Regression - {normalize_method.value}\t        - Accuracy: \t{accuracy:2}")

        for i in [1, 3, 5, 7, 9]:
            model = train_knn(X_train, y_train, i)
            accuracy = evaluate_model(model, X_test, y_test)
            print(f"K Neatest Neighbor  - {normalize_method.value}\t - k: {i} - Accuracy: \t{accuracy:2}")


if __name__ == "__main__":
    main()
