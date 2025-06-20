#!/usr/bin/env python
# coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import magic
import numpy as np
from secml.array import CArray
from secml_malware.models.malconv import MalConv, DNN_Net
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware, End2EndModel
from secml_malware.attack.whitebox.c_headerPlus_evasion import CHeaderPlusEvasion
import matplotlib.pyplot as plt

# === Load Model ===
net = MalConv()
net = CClassifierEnd2EndMalware(net)
net.load_pretrained_model('./secml_malware/data/trained/pretrained_malconv.pth')
partial_dos = CHeaderPlusEvasion(net, random_init=False, iterations=10,
                                  header_and_padding=True, threshold=0.5,
                                  how_many=144, is_debug=False)

# === Load Dataset ===
Train_folder = "/home/subash/Desktop/phase2/Dataset/Virus/Virus train/Zeroaccess"
Test_folder = "/home/subash/Desktop/phase2/Dataset/Virus/Virus test/Zeroaccess"

Train_X, Train_y, train_file_names = [], [], []
Test_X, Test_y, test_file_names = [], [], []

# --- Load Train ---
for f in os.listdir(Train_folder):
    path = os.path.join(Train_folder, f)
    if "PE32" not in magic.from_file(path):
        continue
    with open(path, "rb") as fh:
        code = fh.read()
    x = End2EndModel.bytes_to_numpy(code, net.get_input_max_length(), 256, False)
    _, confidence = net.predict(CArray(x), True)
    if confidence[0, 1].item() < 0.5:
        continue
    Train_X.append(x)
    Train_y.append([1 - confidence[1][0].item(), confidence[1][0].item()])
    train_file_names.append(path)

# --- Load Test ---
for f in os.listdir(Test_folder):
    path = os.path.join(Test_folder, f)
    if "PE32" not in magic.from_file(path):
        continue
    with open(path, "rb") as fh:
        code = fh.read()
    x = End2EndModel.bytes_to_numpy(code, net.get_input_max_length(), 256, False)
    _, confidence = net.predict(CArray(x), True)
    if confidence[0, 1].item() < 0.5:
        continue
    Test_X.append(x)
    Test_y.append([1 - confidence[1][0].item(), confidence[1][0].item()])
    test_file_names.append(path)

# === Clean Test Accuracy ===
clean_correct = 0
for sample in Test_X:
    _, confidence = net.predict(CArray(sample), True)
    if confidence[0, 1].item() >= 0.5:
        clean_correct += 1
clean_test_accuracy = clean_correct / len(Test_y) * 100
print(f"\n✅ Clean Test Accuracy (before attack): {clean_test_accuracy:.2f}%")

# === Defense Function ===
def defense_function(x):
    return np.clip(x, 0, 255)

# === Evasion + Defense Loop ===
evasion_rates_train, evasion_rates_test, defense_accuracies = [], [], []
best_success_rate = 0

for epoch in range(5):
    print(f"\n📘 Epoch {epoch}")
    sample_byte_changes_train = []
    train_success = 0

    # --- Generate patch per train sample ---
    for x, label in zip(Train_X, Train_y):
        _, _, _, f_obj, byte_change = partial_dos.run(CArray(x), CArray(label[1]))
        if f_obj < 0.5:
            train_success += 1
        sample_byte_changes_train.append(byte_change)

    train_patch_success_rate = train_success / len(Train_X)
    evasion_rates_train.append(train_patch_success_rate * 100)
    print(f"📌 Train Evasion Rate (patch gen): {train_patch_success_rate:.3%}")

    # --- Evaluate train with defense ---
    train_defense_success, defense_correct = 0, 0
    for x, label, byte_change in zip(Train_X, Train_y, sample_byte_changes_train):
        x_defended = x.copy()
        indexes_to_perturb = list(range(2, 0x3C))
        padding_positions = CArray(x).find(CArray(x) == 256)
        if padding_positions:
            indexes_to_perturb += list(range(padding_positions[0],
                                             min(len(x), padding_positions[0] + 144)))
        for b in range(min(len(byte_change), len(indexes_to_perturb))):
            x_defended[indexes_to_perturb[b]] = byte_change[b]

        x_defended = defense_function(x_defended)
        _, confidence = net.predict(CArray(x_defended), True)
        if confidence[0, 1].item() < 0.5:
            train_defense_success += 1
        else:
            defense_correct += 1

    avg_train_success = train_defense_success / len(Train_X)
    print(f"🧪 Train Evasion Rate (after defense): {avg_train_success:.3%}")

    # --- Apply attack and defense on test set ---
    test_success, defense_correct_test = 0, 0
    for x, label in zip(Test_X, Test_y):
        _, _, _, f_obj, byte_change = partial_dos.run(CArray(x), CArray(label[1]))

        x_defended = x.copy()
        indexes_to_perturb = list(range(2, 0x3C))
        padding_positions = CArray(x).find(CArray(x) == 256)
        if padding_positions:
            indexes_to_perturb += list(range(padding_positions[0],
                                             min(len(x), padding_positions[0] + 144)))
        for b in range(min(len(byte_change), len(indexes_to_perturb))):
            x_defended[indexes_to_perturb[b]] = byte_change[b]

        x_defended = defense_function(x_defended)
        _, confidence = net.predict(CArray(x_defended), True)
        if confidence[0, 1].item() < 0.5:
            test_success += 1
        else:
            defense_correct_test += 1

    avg_test_success = test_success / len(Test_X)
    test_defense_acc = defense_correct_test / len(Test_X)
    evasion_rates_test.append(avg_test_success * 100)
    defense_accuracies.append(test_defense_acc * 100)

    print(f"🧪 Test Evasion Rate (after defense): {avg_test_success:.3%}")
    print(f"🛡️  Defense Accuracy Rate: {test_defense_acc:.3%}")

    if avg_test_success > best_success_rate:
        best_success_rate = avg_test_success

# === Results Summary ===
epochs = list(range(1, 6))
plt.figure(figsize=(10, 6))
plt.plot(epochs, evasion_rates_train, label="Train Evasion Rate", marker='o')
plt.plot(epochs, evasion_rates_test, label="Test Evasion Rate", marker='o')
plt.plot(epochs, defense_accuracies, label="Defense Accuracy", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Rate (%)")
plt.title("Evasion Rate and Defense Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Final Metrics ---
best_defense_acc = max(defense_accuracies)
best_epoch_defense = defense_accuracies.index(best_defense_acc) + 1
best_test_evasion_rate = max(evasion_rates_test)
best_epoch_test_evasion = evasion_rates_test.index(best_test_evasion_rate) + 1

if clean_test_accuracy > best_defense_acc:
    print("\n✅ Clean accuracy is higher than best defense accuracy.")
else:
    print("\n❌ Defense accuracy exceeded clean accuracy.")

# === Highlighted Plot for Defense Accuracy ===
plt.figure(figsize=(10, 6))
plt.plot(epochs, defense_accuracies, label="Defense Accuracy", marker='o', color='green')
plt.axhline(best_defense_acc, color='red', linestyle='--',
            label=f"Best Defense Acc: {best_defense_acc:.2f}% (Epoch {best_epoch_defense})")
plt.xlabel("Epoch")
plt.ylabel("Defense Accuracy (%)")
plt.title("Defense Accuracy Highlight")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Evasion vs Defense Scatter Plot ===
plt.figure(figsize=(8, 6))
plt.scatter(evasion_rates_test, defense_accuracies, color='purple')
for i, (x, y) in enumerate(zip(evasion_rates_test, defense_accuracies)):
    plt.text(x + 0.5, y + 0.5, f'E{i+1}')
plt.xlabel("Test Evasion Rate (%)")
plt.ylabel("Defense Accuracy (%)")
plt.title("Evasion vs Defense Tradeoff")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Clean vs Defense Accuracy Bar ===
plt.figure(figsize=(8, 6))
plt.bar(['Clean Accuracy', 'Best Defense Accuracy'],
        [clean_test_accuracy, best_defense_acc],
        color=['skyblue', 'salmon'])
plt.title("Clean Accuracy vs Defense Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

print(f"\n📊 Best Defense Accuracy: {best_defense_acc:.2f}% (Epoch {best_epoch_defense})")
print(f"📈 Highest Test Evasion Rate: {best_test_evasion_rate:.2f}% (Epoch {best_epoch_test_evasion})")
print(f"🧪 Clean Test Accuracy (pre-attack): {clean_test_accuracy:.2f}%")
