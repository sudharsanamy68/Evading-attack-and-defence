import os
import numpy as np
from secml.array import CArray
from secml.ml.classifiers import CClassifierSVM
from secml.data import CDataset
from secml.ml.kernels import CKernelLinear

# Define paths
MALWARE_TRAIN_DIR = "/home/subash/Desktop/phase2/Dataset/Virus/Virus train/Locker"
MALWARE_TEST_DIR = "/home/subash/Desktop/phase2/Dataset/Virus/Virus test/Locker"
BENIGN_TRAIN_DIR = "/home/subash/Desktop/phase2/Dataset/Benign/Benign train"
BENIGN_TEST_DIR = "/home/subash/Desktop/phase2/Dataset/Benign/Benign test"

ADV_TRAIN_DIR = "/home/subash/Desktop/phase2/malpatch/MalPatch_binary/adv_train"  # folder of adversarial malware train samples
ADV_TEST_DIR = "/home/subash/Desktop/phase2/malpatch/MalPatch_binary/adv_test"    # folder of adversarial malware test samples

FIXED_LEN = 2**20  # 1MB

def pad_or_truncate(data_bytes, fixed_len=FIXED_LEN):
    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    if len(arr) < fixed_len:
        arr = np.pad(arr, (0, fixed_len - len(arr)), 'constant')
    else:
        arr = arr[:fixed_len]
    return arr

def load_dataset(folder_path, label):
    X, y = [], []
    for fname in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, fname)
        if not os.path.isfile(path):
            continue
        with open(path, 'rb') as f:
            data = f.read()
        arr = pad_or_truncate(data)
        X.append(arr)
        y.append(label)
    return X, y

# Load clean malware and benign data
X_mal_train, y_mal_train = load_dataset(MALWARE_TRAIN_DIR, label=1)
X_mal_test, y_mal_test = load_dataset(MALWARE_TEST_DIR, label=1)

X_benign_train, y_benign_train = load_dataset(BENIGN_TRAIN_DIR, label=0)
X_benign_test, y_benign_test = load_dataset(BENIGN_TEST_DIR, label=0)

# Load adversarial malware samples
X_adv_train, y_adv_train = load_dataset(ADV_TRAIN_DIR, label=1)
X_adv_test, y_adv_test = load_dataset(ADV_TEST_DIR, label=1)

# Combine for defense training and testing
X_train_defense = CArray(X_mal_train + X_adv_train + X_benign_train)
y_train_defense = CArray(y_mal_train + y_adv_train + y_benign_train)

X_test_defense = CArray(X_mal_test + X_adv_test + X_benign_test)
y_test_defense = CArray(y_mal_test + y_adv_test + y_benign_test)

# Train defense model (SVM)
clf = CClassifierSVM(kernel=CKernelLinear(), C=1.0)

print("\n[INFO] Training defense model...")
clf.fit(X_train_defense, y_train_defense)

# Evaluate model
y_pred = clf.predict(X_test_defense)
acc = (y_pred == y_test_defense).sum() / y_test_defense.shape[0]

print(f"\n✅ Defense Accuracy on (Malware+Benign+Adv) Test Set: {acc * 100:.2f}%")

# Evaluate separately for adversarial evasion
acc_adv = (clf.predict(CArray(X_adv_test)) == CArray(y_adv_test)).sum() / len(y_adv_test)
evasion_rate_after = 1.0 - acc_adv
print(f"🛡️ Adversarial Detection Accuracy: {acc_adv * 100:.2f}%")
print(f"🚫 Evasion Rate After Defense: {evasion_rate_after * 100:.2f}%")
