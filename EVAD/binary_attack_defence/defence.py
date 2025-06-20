#!/usr/bin/env python
# coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import magic
import numpy as np
from secml.array import CArray
from secml_malware.models.malconv import MalConv, DNN_Net
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware, End2EndModel
from secml_malware.attack.whitebox.c_headerPlus_evasion import CHeaderPlusEvasion
import matplotlib.pyplot as plt

# Define target model
net_choice = 'MalConv'  # MalConv/AvastNet

if net_choice == 'MalConv':
    net = MalConv()
    net = CClassifierEnd2EndMalware(net)
    net.load_pretrained_model()
    net.load_pretrained_model('./secml_malware/data/trained/pretrained_malconv.pth')
    partial_dos = CHeaderPlusEvasion(net, random_init=False, iterations=10,
                                      header_and_padding=True, threshold=0.5,
                                      how_many=144, is_debug=False)
else:
    net = DNN_Net()
    net = CClassifierEnd2EndMalware(net)
    net.load_pretrained_model('./secml_malware/data/trained/dnn_pe.pth')
    partial_dos = CHeaderPlusEvasion(net, random_init=False, iterations=10,
                                      header_and_padding=False, threshold=0.5,
                                      how_many=0, is_debug=False)

# Load dataset
Train_folder = "/home/subash/Desktop/phase2/Dataset/Virus/Virus train/Locker"
Test_folder = "/home/subash/Desktop/phase2/Dataset/Virus/Virus test/Locker"

Train_X, Train_y, train_file_names = [], [], []
Test_X, Test_y, test_file_names = [], [], []

# Load Train samples
for f in os.listdir(Train_folder):
    path = os.path.join(Train_folder, f)
    if "PE32" not in magic.from_file(path):
        continue
    with open(path, "rb") as file_handle:
        code = file_handle.read()

    x = End2EndModel.bytes_to_numpy(code, net.get_input_max_length(), 256, False)
    _, confidence = net.predict(CArray(x), True)

    if confidence[0, 1].item() < 0.5:
        continue

    print(f"> Added Train: {f} with confidence {confidence[0,1].item()}")
    Train_X.append(x)
    conf = confidence[1][0].item()
    Train_y.append([1 - conf, conf])
    train_file_names.append(path)

# Load Test samples
for f in os.listdir(Test_folder):
    path = os.path.join(Test_folder, f)
    if "PE32" not in magic.from_file(path):
        continue
    with open(path, "rb") as file_handle:
        code = file_handle.read()

    x = End2EndModel.bytes_to_numpy(code, net.get_input_max_length(), 256, False)
    _, confidence = net.predict(CArray(x), True)

    if confidence[0, 1].item() < 0.5:
        continue

    print(f"> Added Test: {f} with confidence {confidence[0,1].item()}")
    Test_X.append(x)
    conf = confidence[1][0].item()
    Test_y.append([1 - conf, conf])
    test_file_names.append(path)

# Simple defense: clipping bytes to [0, 255]
def defense_function(x):
    return np.clip(x, 0, 255)

# For plotting
evasion_rates_train, evasion_rates_test, defense_accuracies = [], [], []
success, best_success_rate = 0, 0

# Evasion + Defense loop
for epoch in range(5):
    success = 0
    for sample, label in zip(Train_X, Train_y):
        y_pred, adv_score, adv_ds, f_obj, byte_change = partial_dos.run(CArray(sample), CArray(label[1]))
        if f_obj < 0.5:
            success += 1
        adv_x = adv_ds.X[0, :]
        real_adv_x = partial_dos.create_real_sample_from_adv(train_file_names[0], adv_x)

    avg_success = success / len(train_file_names)
    evasion_rates_train.append(avg_success * 100)
    print(f"Epoch:{epoch} Train evasion rate (patch gen): {avg_success:.3%}")

    # Apply patch to train set and check evasion
    train_success, defense_correct = 0, 0
    for sample, label in zip(Train_X, Train_y):
        indexes_to_perturb = list(range(2, 0x3C))
        padding_positions = CArray(sample).find(CArray(sample) == 256)
        if padding_positions:
            indexes_to_perturb += list(range(padding_positions[0], min(len(sample), padding_positions[0] + 144)))
        for b in range(len(indexes_to_perturb)):
            sample[indexes_to_perturb[b]] = byte_change[b]
        defended_sample = defense_function(sample)
        _, confidence = net.predict(CArray(defended_sample), True)
        if confidence[0, 1].item() < 0.5:
            train_success += 1
        else:
            defense_correct += 1
    avg_train_success = train_success / len(train_file_names)
    print(f"Epoch:{epoch} Train evasion rate: {avg_train_success:.3%}")

    # Apply patch to test set and evaluate
    test_success, defense_correct_test = 0, 0
    for sample, label in zip(Test_X, Test_y):
        indexes_to_perturb = list(range(2, 0x3C))
        padding_positions = CArray(sample).find(CArray(sample) == 256)
        if padding_positions:
            indexes_to_perturb += list(range(padding_positions[0], min(len(sample), padding_positions[0] + 144)))
        for b in range(len(indexes_to_perturb)):
            sample[indexes_to_perturb[b]] = byte_change[b]
        defended_sample = defense_function(sample)
        _, confidence = net.predict(CArray(defended_sample), True)
        if confidence[0, 1].item() < 0.5:
            test_success += 1
        else:
            defense_correct_test += 1
    avg_test_success = test_success / len(test_file_names)
    evasion_rates_test.append(avg_test_success * 100)
    defense_acc = defense_correct_test / len(test_file_names)
    defense_accuracies.append(defense_acc * 100)

    print(f"Epoch:{epoch} Test evasion rate: {avg_test_success:.3%}")
    print(f"Epoch:{epoch} Defense accuracy rate: {defense_acc:.3%}")

    if avg_test_success > best_success_rate:
        best_success_rate = avg_test_success

print("Best evasion rate on testset: {:.3f}%".format(100 * best_success_rate))

# ===== Plot Results =====
epochs = list(range(1, 6))
plt.figure(figsize=(10, 6))
plt.plot(epochs, evasion_rates_train, label="Train Evasion Rate", marker='o')
plt.plot(epochs, evasion_rates_test, label="Test Evasion Rate", marker='o')
plt.plot(epochs, defense_accuracies, label="Defense Accuracy Rate", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Rate (%)")
plt.title("Evasion Rate and Defense Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
