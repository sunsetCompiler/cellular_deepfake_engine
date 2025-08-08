#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 16:58:33 2025

@author: nirupmasingh
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.utils.spectral_norm as spectral_norm

# Configuration
img_size = 64        # Size of images
z_dim = 256          # Latent space dimension
batch_size = 64      # Batch size for training
learning_rate = 0.0002  # Learning rate (slightly higher)
beta1 = 0.5          # Adam optimizer beta1
beta2 = 0.999        # Adam optimizer beta2
epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# === DATASET AND TRANSFORM ===
# Improved transforms with augmentation for training stability
transform = transforms.Compose([
    transforms.Resize(img_size + 8),  # Resize larger then crop for augmentation
    transforms.RandomCrop(img_size),  # Random crop for data augmentation
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),  # Color jitter
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=15),  # random rotation
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=10),  # affine transform
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # blur to simulate focus variance
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),  # random occlusion

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1] for all channels
])

dataset = ImageFolder(root='kaggle_bloodcell', transform=transform)
num_classes = len(dataset.classes)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

print(f"Found classes: {dataset.classes}")
print(f"Number of classes: {num_classes}")
print(f"Dataset size: {len(dataset)} images")

# Weight initialization function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
        
        
    # === GENERATOR ===
class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, img_channels=3, ngf=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.label_emb = nn.Embedding(num_classes, z_dim)
        
        # Project and reshape
        self.project = nn.Sequential(
            nn.Linear(z_dim * 2, ngf * 8 * 4 * 4),
            nn.BatchNorm1d(ngf * 8 * 4 * 4),
            nn.ReLU(True)
        )
        
        # Main generator structure using transposed convolutions
        self.main = nn.Sequential(
            # State size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Output state size: (img_channels) x 64 x 64
        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        x = torch.cat((noise, label_embedding), dim=1)
        x = self.project(x)
        x = x.view(x.size(0), -1, 4, 4)
        return self.main(x)
    
    
# === DISCRIMINATOR ===
class Discriminator(nn.Module):
    def __init__(self, num_classes, img_channels=3, ndf=64):
        super(Discriminator, self).__init__()
        
        # Process label for conditioning
        self.label_embed = nn.Embedding(num_classes, img_size * img_size)
        
        # Image processing path
        self.img_path = nn.Sequential(
            # Input size: (img_channels) x 64 x 64
            spectral_norm(nn.Conv2d(img_channels + 1, ndf, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.3, inplace=True),
            # State size: (ndf) x 32 x 32
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LayerNorm([ndf * 2, 16, 16]),  # Replaces BatchNorm2d
            nn.LeakyReLU(0.3, inplace=True),
            
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LayerNorm([ndf * 4, 8, 8]),
            nn.LeakyReLU(0.3, inplace=True),
            
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.LayerNorm([ndf * 8, 4, 4]),
            nn.LeakyReLU(0.3, inplace=True),
            
            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )

    def forward(self, img, labels):
        batch_size = img.size(0)
        label_embedding = self.label_embed(labels).view(batch_size, 1, img_size, img_size)
        x = torch.cat((img, label_embedding), dim=1)
        return self.img_path(x).view(batch_size, -1)
    
    
# === INITIALIZATION ===
generator = Generator(z_dim, num_classes).to(device)
discriminator = Discriminator(num_classes).to(device)

# Initialize weights
generator.apply(weights_init)
discriminator.apply(weights_init)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

# Create fixed noise for evaluation
fixed_noise = torch.randn(16, z_dim, device=device)
fixed_labels = torch.tensor([i % num_classes for i in range(16)], device=device)

# Lists to track progress
G_losses = []
D_losses = []


# === TRAINING LOOP ===
print("Starting training...")

for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        batch_size_curr = imgs.size(0)

        # === Train Discriminator ===
        # Real images
        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size_curr, 1).to(device) * 0.9  # Label smoothing
        d_output_real = discriminator(imgs, labels)
        d_loss_real = criterion(d_output_real, real_labels)

        # Fake images
        noise = torch.randn(batch_size_curr, z_dim).to(device)
        fake_labels = torch.randint(0, num_classes, (batch_size_curr,), device=device)
        fake_imgs = generator(noise, fake_labels)
        fake_output = discriminator(fake_imgs.detach(), fake_labels)
        fake_targets = torch.zeros(batch_size_curr, 1).to(device)
        d_loss_fake = criterion(fake_output, fake_targets)

        # Combined loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # === Train Generator ===
        optimizer_G.zero_grad()
        # Create new noise and labels
        noise = torch.randn(batch_size_curr, z_dim).to(device)
        gen_labels = torch.randint(0, num_classes, (batch_size_curr,), device=device)
        fake_imgs = generator(noise, gen_labels)
        
        # Try to fool the discriminator
        validity = discriminator(fake_imgs, gen_labels)
        g_loss = criterion(validity, torch.ones(batch_size_curr, 1).to(device))

        g_loss.backward()
        optimizer_G.step()

        # Store losses for plotting
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

    # Print status
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
        
        # Generate and save sample images
        with torch.no_grad():
            generator.eval()
            fake_imgs = generator(fixed_noise, fixed_labels)
            fake_imgs = (fake_imgs + 1) / 2  # Unnormalize from [-1,1] to [0,1]
            
            # Save the images
            os.makedirs("generated_images", exist_ok=True)
            vutils.save_image(fake_imgs, f"generated_images/epoch_{epoch+1}.png", 
                            nrow=4, normalize=False)
            print(f"Saved generated_images/epoch_{epoch+1}.png")
            generator.train()

print("Training complete!")

# Save the trained models
torch.save(generator.state_dict(), "improved_generator.pth")
torch.save(discriminator.state_dict(), "improved_discriminator.pth")
print("Models saved!")

# Plot losses
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="Generator", alpha=0.7)
plt.plot(D_losses, label="Discriminator", alpha=0.7)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_plot.png")
plt.show()

# Generate final sample images for each class
print("Generating final sample images for each class...")
generator.eval()
with torch.no_grad():
    samples_per_class = 20
    num_rows = num_classes
    num_cols = samples_per_class // 4
    all_fake_imgs = []
    
    for class_idx in range(num_classes):
        class_noise = torch.randn(samples_per_class, z_dim, device=device)
        class_labels = torch.full((samples_per_class,), class_idx, device=device)
        fake_class_imgs = generator(class_noise, class_labels)
        fake_class_imgs = (fake_class_imgs + 1) / 2  # Unnormalize
        all_fake_imgs.append(fake_class_imgs)
    
    # Combine all images
    all_fake_imgs = torch.cat(all_fake_imgs)
    
    # Save grid of images
    vutils.save_image(all_fake_imgs, "final_samples.png", 
                     nrow=samples_per_class, normalize=False)
    print("Final samples saved to final_samples.png")
    
    
# Generate synthetic dataset for data augmentation
print("Generating synthetic images for data augmentation...")
os.makedirs("augmented_dataset", exist_ok=True)
generator.eval()

with torch.no_grad():
    for class_idx, class_name in enumerate(dataset.classes):
        os.makedirs(f"augmented_dataset/{class_name}", exist_ok=True)
        print(f"Generating images for class: {class_name}")
        
        for i in range(100):  # Generate 100 images per class
            z = torch.randn(1, z_dim, device=device)
            labels = torch.tensor([class_idx], device=device)
            fake_img = generator(z, labels)
            
            # Unnormalize
            fake_img = (fake_img + 1) / 2
            
            # Save the image
            save_path = f"augmented_dataset/{class_name}/{i+1:03d}.png"
            vutils.save_image(fake_img, save_path, normalize=False)
        
        print(f"Generated 100 images for class {class_name}")

print("All synthetic images saved to augmented_dataset/")


# === SAVE REAL IMAGES FOR EVALUATION ===
print("Saving a subset of real images for evaluation...")
os.makedirs("real_dataset", exist_ok=True)
for class_name in dataset.classes:
    os.makedirs(f"real_dataset/{class_name}", exist_ok=True)

real_saved = {cls: 0 for cls in dataset.classes}
real_limit = 100  # Save 100 real images per class

for img, label in dataset:
    class_name = dataset.classes[label]
    if real_saved[class_name] < real_limit:
        save_path = f"real_dataset/{class_name}/{real_saved[class_name]+1:03d}.png"
        vutils.save_image(img, save_path, normalize=False)
        real_saved[class_name] += 1
    if all(x >= real_limit for x in real_saved.values()):
        break

print("Real images saved for evaluation.")

# === EVALUATION: SSIM and FID ===
print("🧪 Running evaluation using SSIM and FID...")

from skimage.metrics import structural_similarity as ssim
from PIL import Image
from pytorch_fid import fid_score

def compute_avg_ssim(real_dir, fake_dir, max_images=100):
    real_imgs = sorted(os.listdir(real_dir))[:max_images]
    fake_imgs = sorted(os.listdir(fake_dir))[:max_images]
    scores = []

    for r_name, f_name in zip(real_imgs, fake_imgs):
        real_img = np.array(Image.open(os.path.join(real_dir, r_name)).resize((64, 64))) / 255.0
        fake_img = np.array(Image.open(os.path.join(fake_dir, f_name)).resize((64, 64))) / 255.0

        if real_img.shape != fake_img.shape:
            continue

        if real_img.ndim == 3:
            s = ssim(real_img, fake_img, multichannel=True, data_range=1.0)
        else:
            s = ssim(real_img, fake_img, data_range=1.0)

        scores.append(s)

    return np.mean(scores)

# Compute SSIM for each class (average over all)
all_ssim_scores = []

for class_name in dataset.classes:
    real_dir = f"real_dataset/{class_name}"
    fake_dir = f"augmented_dataset/{class_name}"
    score = compute_avg_ssim(real_dir, fake_dir)
    all_ssim_scores.append(score)
    print(f"SSIM for class '{class_name}': {score:.4f}")

print(f"Average SSIM: {np.mean(all_ssim_scores):.4f}")

# === FID ===
print("Calculating FID...")
fid_value = fid_score.calculate_fid_given_paths(
    paths=["real_dataset", "augmented_dataset"],
    batch_size=32,
    device=device,
    dims=2048
)
print(f"FID Score: {fid_value:.2f}")
