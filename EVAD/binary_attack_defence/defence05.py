import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from secml_malware.models.malconv import MalConv
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Configurations
FIXED_LEN = 2 ** 20  # 1 MB
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4

# Paths
MALWARE_TRAIN = "/home/subash/Desktop/phase2/Dataset/Virus/Virus train/Locker"
MALWARE_TEST = "/home/subash/Desktop/phase2/Dataset/Virus/Virus test/Locker"
BENIGN_TRAIN = "/home/subash/Desktop/phase2/Dataset/Benign/Benign train"
BENIGN_TEST = "/home/subash/Desktop/phase2/Dataset/Benign/Benign test"
ADV_TRAIN = "/home/subash/Desktop/phase2/malpatch/MalPatch_binary/adv_train"
ADV_TEST = "/home/subash/Desktop/phase2/malpatch/MalPatch_binary/adv_test"

# Utility to pad and load binary files
def load_bytes(path):
    with open(path, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    if len(arr) < FIXED_LEN:
        arr = np.pad(arr, (0, FIXED_LEN - len(arr)), 'constant')
    else:
        arr = arr[:FIXED_LEN]
    return arr

# Dataset wrapper
class MalwareDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        arr = load_bytes(self.file_paths[idx])
        return torch.tensor(arr, dtype=torch.long), self.labels[idx]

# Collect data
def gather_data(mal_dir, benign_dir, adv_dir=None):
    X, y = [], []

    if mal_dir:
        for f in os.listdir(mal_dir):
            X.append(os.path.join(mal_dir, f))
            y.append(1)
    if benign_dir:
        for f in os.listdir(benign_dir):
            X.append(os.path.join(benign_dir, f))
            y.append(0)
    if adv_dir:
        for f in os.listdir(adv_dir):
            X.append(os.path.join(adv_dir, f))
            y.append(1)
    return X, y

# Data loaders
train_paths, train_labels = gather_data(MALWARE_TRAIN, BENIGN_TRAIN, ADV_TRAIN)
test_paths, test_labels = gather_data(MALWARE_TEST, BENIGN_TEST, ADV_TEST)
adv_test_paths, adv_test_labels = gather_data(None, None, ADV_TEST)

train_loader = DataLoader(MalwareDataset(train_paths, train_labels), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(MalwareDataset(test_paths, test_labels), batch_size=1, shuffle=False)
adv_loader = DataLoader(MalwareDataset(adv_test_paths, adv_test_labels), batch_size=1, shuffle=False)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
malconv_model = MalConv()
clf = CClassifierEnd2EndMalware(malconv_model)
clf.load_pretrained_model()
net = clf.model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# --- Pre-defense evaluation ---
net.eval()
adv_correct_pre = 0
for x, y in adv_loader:
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        output = net(x)
        pred = output.argmax(dim=1)
        if pred == y:
            adv_correct_pre += 1

adv_acc_pre = adv_correct_pre / len(adv_loader)
evasion_pre = 1 - adv_acc_pre
print(f"\n⚠️ Pre-Defense Adversarial Accuracy: {adv_acc_pre * 100:.2f}%")
print(f"🚫 Pre-Defense Evasion Rate: {evasion_pre * 100:.2f}%")

# --- Adversarial training ---
print("\n[INFO] Starting Adversarial Training...")
net.train()
losses = []
for epoch in range(EPOCHS):
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = net(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

torch.save(net.state_dict(), "./fine_tuned_malconv.pth")
print("[INFO] Fine-tuned model saved.")

# --- Post-defense evaluation ---
net.eval()
correct, total = 0, 0
adv_correct_post = 0
for x, y in test_loader:
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        output = net(x)
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += 1

for x, y in adv_loader:
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        output = net(x)
        pred = output.argmax(dim=1)
        if pred == y:
            adv_correct_post += 1

acc_post = correct / total
adv_acc_post = adv_correct_post / len(adv_loader)
evasion_post = 1 - adv_acc_post

print(f"\n✅ Post-Defense Accuracy on Total Test Set: {acc_post * 100:.2f}%")
print(f"🛡️ Post-Defense Adversarial Accuracy: {adv_acc_post * 100:.2f}%")
print(f"🚫 Post-Defense Evasion Rate: {evasion_post * 100:.2f}%")

# --- Visualization ---
labels = ['Pre-Defense', 'Post-Defense']
evasion_rates = [evasion_pre * 100, evasion_post * 100]
adv_acc_rates = [adv_acc_pre * 100, adv_acc_post * 100]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
bar1 = ax.bar(x - width/2, adv_acc_rates, width, label='Adversarial Accuracy')
bar2 = ax.bar(x + width/2, evasion_rates, width, label='Evasion Rate')

ax.set_ylabel('Percentage')
ax.set_title('Defense Performance vs Evasion')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for bar in bar1 + bar2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.show()
