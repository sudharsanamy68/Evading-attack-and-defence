import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import argparse

# Assuming helper functions like transform, mask creation, and other utilities are already in place

# Add argument parser for command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Patch attack on neural networks")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_iteration', type=int, default=100, help='Max iterations for patch optimization')
    parser.add_argument('--probability_threshold', type=float, default=0.95, help='Probability threshold to stop attack')
    parser.add_argument('--target', type=int, default=123, help='Target class for the attack')  # Default target
    parser.add_argument('--data_train_dir', type=str, default='/home/subash/Desktop/phase2/Dataset_image/Benign_img/Benign_img_train', help="dir of the dataset")
    parser.add_argument('--data_test_dir', type=str, default='/home/subash/Desktop/phase2/Dataset_image/Benign_img/Benign_img_test', help="dir of the dataset")
    parser.add_argument('--model_path', type=str, default='/home/subash/Desktop/phase2/malpatch/MalPatch_binary/secml_malware/data/trained/squeezenet1_0-b66bff10.pth', help="path of the target model")
    parser.add_argument('--pad_row', type=int, default=32, help='Padding row size for the image')
    
    return parser.parse_args()

args = parse_args()

# Function to visualize patch with heatmap
def plot_patch_heatmap(patch):
    patch_np = patch.detach().cpu().numpy()

    # Plot heatmap for each channel (Red, Green, Blue)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i in range(3):
        ax = axes[i]
        sns.heatmap(patch_np[i, :, :], ax=ax, cmap='viridis', cbar=False)
        ax.set_title(f"Channel {i+1} Heatmap")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Function for patch attack
def patch_attack(image, pad_trans, mask, target, probability_threshold, model, lr=1, max_iteration=100):
    model.eval()
    target_probability, count = 0, 0
    g = 0
    perturbated_image = torch.mul(mask.type(torch.FloatTensor), pad_trans.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
    
    while target_probability < probability_threshold and count < max_iteration:
        count += 1
        perturbated_image = torch.autograd.Variable(perturbated_image.data, requires_grad=True)
        per_image = perturbated_image
        per_image = per_image.cuda()
        output = model(per_image)
        target_log_softmax = torch.nn.functional.log_softmax(output, dim=1)[0][target]
        target_log_softmax.backward()
        patch_grad = perturbated_image.grad.clone().cpu()
        perturbated_image.grad.data.zero_()

        g = g + patch_grad / torch.norm(patch_grad, p=1)
        pad_trans = pad_trans.type(torch.FloatTensor) + lr * torch.sign(g)
        pad_trans = torch.clamp(pad_trans, min=0, max=1)
        
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), pad_trans.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image = torch.clamp(perturbated_image, min=0, max=1)
        perturbated_image = perturbated_image.cuda()
        output = model(perturbated_image)
        target_probability = torch.nn.functional.softmax(output, dim=1).data[0][target]
    
    perturbated_image = perturbated_image.cpu().numpy()
    return perturbated_image, pad_trans

# Function to plot success rates over epochs
def plot_success_rates(train_success_rates, test_success_rates):
    plt.figure(figsize=(10, 6))
    plt.plot(range(args.epochs), train_success_rates, label='Train Success Rate', color='blue', marker='o')
    plt.plot(range(args.epochs), test_success_rates, label='Test Success Rate', color='green', marker='o')

    plt.title("Patch Attack Success Rate Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Success Rate (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main training loop
def main():
    # Initialize model
    model = torch.load(args.model_path, map_location='cuda:0')  # Load model on GPU or CPU
    model.eval()

    # Define transforms for image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Assuming mask creation and data loaders are handled properly
    mask = torch.ones((3, 224, 224))  # Example mask (you can create it differently)
    train_loader = DataLoader()  # Your train data loader
    test_loader = DataLoader()  # Your test data loader

    # Track success rates
    train_success_rates = []
    test_success_rates = []

    for epoch in range(args.epochs):
        train_success, train_total, train_actual_total = 0, 0, 0
        
        for (image, label) in train_loader:
            train_total += label.shape[0]
            image = image.cuda()
            label = label.cuda()
            output = model(image)
            _, predicted = torch.max(output.data, 1)

            if predicted[0] == label and predicted[0].data.cpu().numpy() != args.target:
                train_actual_total += 1

                ori_x = np.asarray((image.cpu()).squeeze())
                pad_image = np.pad(ori_x[0], ((0, args.pad_row), (0, 0)), constant_values=0)
                pad_image = Image.fromarray(np.uint8(pad_image * 255))
                pad_trans = transform(pad_image)
                pad_trans = pad_trans.unsqueeze(0).to(device)

                pad_trans = torch.mul(mask.type(torch.FloatTensor), pad_trans.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), pad_trans.type(torch.FloatTensor))

                perturbated_image, applied_patch = patch_attack(image, pad_trans, mask, args.target, args.probability_threshold, model, args.lr, args.max_iteration)
                perturbated_image = torch.from_numpy(perturbated_image).cuda()
                output = model(perturbated_image)
                _, predicted = torch.max(output.data, 1)
                if predicted[0].data.cpu().numpy() == args.target:
                    train_success += 1
                patch = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor))

        train_success_rates.append(train_success / train_actual_total)
        test_success_rate = test_patch(args.pad_row, args.target, patch, test_loader, model, mask)
        test_success_rates.append(test_success_rate)

        # Plot heatmap of the patch
        plot_patch_heatmap(patch)

        print(f"Epoch: {epoch} | Train success rate: {100 * train_success / train_actual_total:.3f}%")
        print(f"Epoch: {epoch} | Test success rate: {100 * test_success_rate:.3f}%")

    # Plot success rates at the end
    plot_success_rates(train_success_rates, test_success_rates)

# Assuming patching and other utilities are implemented correctly
def test_patch(pad_row, target, patch, test_loader, model, mask):
    # Your testing code to evaluate the patch attack on the test set
    pass

if __name__ == "__main__":
    main()
