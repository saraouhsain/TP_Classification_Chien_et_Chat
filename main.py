import os
import cv2
import numpy as np
import warnings

from skimage.feature import hog, local_binary_pattern

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")


# =====================================================
# CONFIGURATION
# =====================================================

IMG_SIZE = 64

TRAIN_DOGS = "dataset/training_set/dogs"
TRAIN_CATS = "dataset/training_set/cats"

TEST_DOGS = "dataset/test_set/dogs"
TEST_CATS = "dataset/test_set/cats"


# =====================================================
# LOAD DATASET
# =====================================================

train_images = []
train_labels = []

test_images = []
test_labels = []


def load_images(folder, label, image_list, label_list):

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path)

        if img is not None:

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            image_list.append(img)

            label_list.append(label)


print("Loading training images...")

load_images(TRAIN_DOGS, 1, train_images, train_labels)
load_images(TRAIN_CATS, 0, train_images, train_labels)

print("Loading testing images...")

load_images(TEST_DOGS, 1, test_images, test_labels)
load_images(TEST_CATS, 0, test_images, test_labels)

print("Train images :", len(train_images))
print("Test images :", len(test_images))


# =====================================================
# COMMON TRAIN FUNCTION
# =====================================================

def train_and_evaluate(X_train, y_train, X_test, y_test, name):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )

    print(f"\nTraining {name} ...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"{name} Accuracy: {acc:.4f}")

    return acc


# =====================================================
# HOG
# =====================================================

print("\n===== HOG =====")

hog_train = []
hog_test = []

for img in train_images:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    f = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2-Hys'
    )

    hog_train.append(f)

for img in test_images:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    f = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2-Hys'
    )

    hog_test.append(f)

hog_acc = train_and_evaluate(
    np.array(hog_train),
    np.array(train_labels),
    np.array(hog_test),
    np.array(test_labels),
    "HOG"
)


# =====================================================
# LBP
# =====================================================

print("\n===== LBP =====")

lbp_train = []
lbp_test = []

radius = 3
n_points = 8 * radius

for img in train_images:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    lbp_train.append(hist)

for img in test_images:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    lbp_test.append(hist)

lbp_acc = train_and_evaluate(
    np.array(lbp_train),
    np.array(train_labels),
    np.array(lbp_test),
    np.array(test_labels),
    "LBP"
)


# =====================================================
# SIFT
# =====================================================

print("\n===== SIFT =====")

sift = cv2.SIFT_create()

sift_train = []
sift_test = []

for img in train_images:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, des = sift.detectAndCompute(gray, None)

    if des is not None:
        f = np.mean(des, axis=0)
    else:
        f = np.zeros(128)

    sift_train.append(f)

for img in test_images:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, des = sift.detectAndCompute(gray, None)

    if des is not None:
        f = np.mean(des, axis=0)
    else:
        f = np.zeros(128)

    sift_test.append(f)

sift_acc = train_and_evaluate(
    np.array(sift_train),
    np.array(train_labels),
    np.array(sift_test),
    np.array(test_labels),
    "SIFT"
)


# =====================================================
# FINAL RESULTS
# =====================================================

print("\n================ FINAL RESULTS ================")

print(f"HOG  : {hog_acc:.4f}")
print(f"LBP  : {lbp_acc:.4f}")
print(f"SIFT : {sift_acc:.4f}")

best = max({"HOG": hog_acc, "LBP": lbp_acc, "SIFT": sift_acc}, key=lambda k: {
    "HOG": hog_acc, "LBP": lbp_acc, "SIFT": sift_acc
}[k])

print("\nBest Method:", best)