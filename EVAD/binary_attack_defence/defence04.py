#!/usr/bin/env python
# coding: utf-8
import os
import magic
import numpy as np
from secml.array import CArray

from secml_malware.attack.blackbox.c_wrapper_phi import CEnd2EndWrapperPhi
from secml_malware.attack.blackbox.ga.c_base_genetic_engine import CGeneticAlgorithm
from secml_malware.models.malconv import MalConv
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware, End2EndModel
from secml_malware.attack.blackbox.c_blackbox_malpatch import CBlackBoxMalPatchProblem

# Paths to clean and adversarial data
TRAIN_FOLDER = "/home/subash/Desktop/phase2/Dataset/Virus/Virus train/Mediyes"
TEST_FOLDER = "/home/subash/Desktop/phase2/Dataset/Virus/Virus test/Mediyes"
ADV_TRAIN_DIR = "./adv_train/Mediyes"
ADV_TEST_DIR = "./adv_test/Mediyes"
os.makedirs(ADV_TRAIN_DIR, exist_ok=True)
os.makedirs(ADV_TEST_DIR, exist_ok=True)

# Load MalConv
net = MalConv()
net = CClassifierEnd2EndMalware(net)
net.load_pretrained_model('./secml_malware/data/trained/pretrained_malconv.pth')
net = CEnd2EndWrapperPhi(net)

# Set up attack
attack = CBlackBoxMalPatchProblem(net, population_size=30, iterations=100, is_debug=False)
engine = CGeneticAlgorithm(attack)

Train_X, Train_y, train_file_names, adv_train_labels = [], [], [], []

# --- Generate adversarial samples on training set ---
print(">>> Generating adversarial training samples...")
for i, f in enumerate(os.listdir(TRAIN_FOLDER)):
    path = os.path.join(TRAIN_FOLDER, f)
    if "PE32" not in magic.from_file(path):
        continue
    with open(path, "rb") as fh:
        code = fh.read()
    x = CArray(np.frombuffer(code, dtype=np.uint8)).atleast_2d()
    _, conf = net.predict(x, True)

    if conf[0, 1].item() < 0.5:
        continue

    Train_X.append(code)
    Train_y.append([1 - conf[1][0].item(), conf[1][0].item()])
    train_file_names.append(path)

    # Generate adversarial sample
    _, _, adv_ds, _, _ = engine.run(x, CArray([conf[1][0].item()]))
    adv_sample = adv_ds.X[0, :]
    adv_path = os.path.join(ADV_TRAIN_DIR, f"adv_train_{i}.exe")
    engine.write_adv_to_file(adv_sample, adv_path)
    adv_train_labels.append([1 - conf[1][0].item(), conf[1][0].item()])

# --- Generate adversarial samples on test set ---
print(">>> Generating adversarial test samples...")
Test_y, test_file_names, adv_test_labels = [], [], []
for i, f in enumerate(os.listdir(TEST_FOLDER)):
    path = os.path.join(TEST_FOLDER, f)
    if "PE32" not in magic.from_file(path):
        continue
    with open(path, "rb") as fh:
        code = fh.read()
    x = CArray(np.frombuffer(code, dtype=np.uint8)).atleast_2d()
    _, conf = net.predict(x, True)

    if conf[0, 1].item() < 0.5:
        continue

    Test_y.append([1 - conf[1][0].item(), conf[1][0].item()])
    test_file_names.append(path)

    # Generate adversarial sample
    _, _, adv_ds, _, _ = engine.run(x, CArray([conf[1][0].item()]))
    adv_sample = adv_ds.X[0, :]
    adv_path = os.path.join(ADV_TEST_DIR, f"adv_test_{i}.exe")
    engine.write_adv_to_file(adv_sample, adv_path)
    adv_test_labels.append([1 - conf[1][0].item(), conf[1][0].item()])

# --- Adversarial Training: Combine Clean + Adversarial Training Samples ---
print(">>> Performing adversarial training...")
adv_model = MalConv()
defense_net = CClassifierEnd2EndMalware(adv_model)

X_defense, y_defense = [], []

# Add clean training samples
for data, label in zip(Train_X, Train_y):
    X_defense.append(CArray(np.frombuffer(data, dtype=np.uint8)))
    y_defense.append(CArray(label))

# Add adversarial training samples
for fname, label in zip(sorted(os.listdir(ADV_TRAIN_DIR)), adv_train_labels):
    with open(os.path.join(ADV_TRAIN_DIR, fname), 'rb') as f:
        data = f.read()
    X_defense.append(CArray(np.frombuffer(data, dtype=np.uint8)))
    y_defense.append(CArray(label))

# Train (fine-tune) model
defense_net.fit(X_defense, y_defense)
defense_net.save_pretrained_model('./defended_malconv.pth')

# --- Load defense model ---
print(">>> Evaluating defense model...")
defense_net = MalConv()
defense_net = CClassifierEnd2EndMalware(defense_net)
defense_net.load_pretrained_model('./defended_malconv.pth')

# --- Evaluate on adversarial test samples ---
test_success = 0
for fname in sorted(os.listdir(ADV_TEST_DIR)):
    with open(os.path.join(ADV_TEST_DIR, fname), 'rb') as f:
        code = f.read()
    x = CArray(np.frombuffer(code, dtype=np.uint8)).atleast_2d()
    _, conf = defense_net.predict(x, True)

    if conf[0, 1].item() < 0.5:
        test_success += 1  # Still evaded detection

evasion_after_defense = test_success / len(os.listdir(ADV_TEST_DIR))
print("\n===============================")
print(f"⚠️  Evasion rate after defense: {evasion_after_defense*100:.2f}%")
print(f"✅ Defense accuracy: {(1 - evasion_after_defense)*100:.2f}%")
print("===============================")
