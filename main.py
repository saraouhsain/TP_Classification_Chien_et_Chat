import os
import cv2
import numpy as np
import warnings

from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")


# CONFIGURATION


IMG_SIZE = 128

TRAIN_DOGS = "dataset/training_set/dogs"
TRAIN_CATS = "dataset/training_set/cats"
TEST_DOGS  = "dataset/test_set/dogs"
TEST_CATS  = "dataset/test_set/cats"


# LOAD DATASET


def load_images(folder, label, image_list, label_list):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img  = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            image_list.append(img)
            label_list.append(label)

train_images, train_labels = [], []
test_images,  test_labels  = [], []

print("Loading training images...")
load_images(TRAIN_DOGS, 1, train_images, train_labels)
load_images(TRAIN_CATS, 0, train_images, train_labels)

print("Loading testing images...")
load_images(TEST_DOGS, 1, test_images, test_labels)
load_images(TEST_CATS, 0, test_images, test_labels)

print(f"Train : {len(train_images)} images")
print(f"Test  : {len(test_images)} images")


# FEATURE EXTRACTORS


# ---------- HOG ----------
def extract_hog(images):
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        f = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        features.append(f)
    return np.array(features)

# ---------- LBP ----------
def extract_lbp(images):
    features = []
    radius   = 3
    n_points = 8 * radius
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp  = local_binary_pattern(gray, n_points, radius, method="uniform")
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, n_points + 3),
            range=(0, n_points + 2)
        )
        hist = hist.astype("float") / (hist.sum() + 1e-6)
        features.append(hist)
    return np.array(features)

# ---------- SIFT ----------
def extract_sift(images):
    sift     = cv2.SIFT_create()
    features = []
    for img in images:
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, des = sift.detectAndCompute(gray, None)
        if des is not None:
            f = np.concatenate([
                np.mean(des, axis=0),
                np.std(des,  axis=0),
                np.max(des,  axis=0)
            ])
        else:
            f = np.zeros(384)
        features.append(f)
    return np.array(features)

# ---------- SURF (ORB fallback si opencv-contrib absent) ----------
def extract_surf(images):
    # Tenter SURF, sinon ORB
    try:
        detector = cv2.xfeatures2d.SURF_create(400)
        desc_size = 64
        print("  Using SURF")
    except Exception:
        detector = cv2.ORB_create(nfeatures=500)
        desc_size = 32
        print("  SURF unavailable → using ORB as fallback")

    features = []
    for img in images:
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, des = detector.detectAndCompute(gray, None)
        if des is not None:
            des = des.astype("float32")
            f = np.concatenate([
                np.mean(des, axis=0),
                np.std(des,  axis=0),
                np.max(des,  axis=0)
            ])
        else:
            f = np.zeros(desc_size * 3)
        features.append(f)
    return np.array(features)

# TRAIN + EVALUATE  →  retourne (acc, model, scaler)

def train_and_evaluate(X_train, y_train, X_test, y_test, name):
    scaler = StandardScaler()
    Xtr    = scaler.fit_transform(X_train)
    Xte    = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )

    print(f"  Training MLP on {name} features...")
    model.fit(Xtr, y_train)

    y_pred = model.predict(Xte)
    acc    = accuracy_score(y_test, y_pred)
    print(f"  {name} Accuracy : {acc:.4f}")

    return acc, model, scaler

# EXTRACTION + EVALUATION PAR METHODE

results = {}

print("\n        HOG ")
hog_tr = extract_hog(train_images)
hog_te = extract_hog(test_images)
acc, model, scaler = train_and_evaluate(hog_tr, train_labels, hog_te, test_labels, "HOG")
results["HOG"] = (acc, model, scaler, hog_tr, hog_te)

print("\n       LBP     ")
lbp_tr = extract_lbp(train_images)
lbp_te = extract_lbp(test_images)
acc, model, scaler = train_and_evaluate(lbp_tr, train_labels, lbp_te, test_labels, "LBP")
results["LBP"] = (acc, model, scaler, lbp_tr, lbp_te)

print("\n   SIFT  ")
sift_tr = extract_sift(train_images)
sift_te = extract_sift(test_images)
acc, model, scaler = train_and_evaluate(sift_tr, train_labels, sift_te, test_labels, "SIFT")
results["SIFT"] = (acc, model, scaler, sift_tr, sift_te)

print("\n       SURF       ")
surf_tr = extract_surf(train_images)
surf_te = extract_surf(test_images)
acc, model, scaler = train_and_evaluate(surf_tr, train_labels, surf_te, test_labels, "SURF")
results["SURF"] = (acc, model, scaler, surf_tr, surf_te)

# =====================================================
# COMBINED  (HOG + LBP + SIFT + SURF)


print("\n       COMBINED (HOG + LBP + SIFT + SURF) ")
combined_tr = np.concatenate([hog_tr, lbp_tr, sift_tr, surf_tr], axis=1)
combined_te = np.concatenate([hog_te, lbp_te, sift_te, surf_te], axis=1)
acc, model, scaler = train_and_evaluate(combined_tr, train_labels, combined_te, test_labels, "COMBINED")
results["COMBINED"] = (acc, model, scaler, combined_tr, combined_te)


# FINAL RESULTS


print("\n                 FINAL RESULTS     ")
for name, (acc, *_) in results.items():
    bar  = "█" * int(acc * 40)
    print(f"{name:<10} {acc:.4f}  {bar}")

best_name = max(results, key=lambda k: results[k][0])
best_acc, best_model, best_scaler, best_tr, best_te = results[best_name]
print(f"\nBest Method : {best_name} ({best_acc:.4f})")

# Rapport détaillé du meilleur modèle
Xte_scaled = best_scaler.transform(best_te)
y_pred     = best_model.predict(Xte_scaled)
print(f"\nClassification Report ({best_name}):")
print(classification_report(test_labels, y_pred, target_names=["Cat", "Dog"]))

print("Confusion Matrix:")
print(confusion_matrix(test_labels, y_pred))


# TEST SUR UNE IMAGE


print("\n            TEST PHASE            ")
print(f"(Using best method : {best_name})")

image_path = input("\nEnter image path : ")
img = cv2.imread(image_path)

if img is None:
    print("Image not found!")
else:
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Extraire avec la même méthode que le meilleur modèle
    extractor_map = {
        "HOG":      extract_hog,
        "LBP":      extract_lbp,
        "SIFT":     extract_sift,
        "SURF":     extract_surf,
        "COMBINED": lambda imgs: np.concatenate([
            extract_hog(imgs),
            extract_lbp(imgs),
            extract_sift(imgs),
            extract_surf(imgs)
        ], axis=1)
    }

    feature     = extractor_map[best_name]([img_resized])
    feature     = best_scaler.transform(feature)

    prediction  = best_model.predict(feature)[0]
    probability = best_model.predict_proba(feature)[0]

    dog_prob = probability[1] * 100
    cat_prob = probability[0] * 100

    print("\n       RESULT        ")
    print(f"Prediction     : {'DOG' if prediction == 1 else 'CAT'}")
    print(f"Dog confidence : {dog_prob:.2f}%")
    print(f"Cat confidence : {cat_prob:.2f}%")

    cv2.imshow("Test Image", img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()