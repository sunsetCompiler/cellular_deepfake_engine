# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 15:57:22 2025

@author: nirup
"""

"""
wgan_bbbc018.py
========================================
Self-Attention WGAN-GP on BBBC018 outlines
• 128×128 resolution (fits any GPU; stable)
• InstanceNorm2d (no LayerNorm shape issues)
• 800 epochs, 5-critic steps, LR-decay
• Saves real/fake samples, reports SSIM & FID
Prereqs: pip install torch torchvision pillow tqdm scikit-image pytorch-fid
"""

import os, numpy as np, torch, torch.nn as nn
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils as vutils
from tqdm import tqdm
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from pytorch_fid import fid_score

# ---------- paths (edit!) ----------
DATA_PATH = r"C:\Akshaj\Desktop\Research Project\datasets\BBBC018_v1_outlines"   # real PNGs inside one class folder
OUT_DIR   = r"C:\Akshaj\Desktop\Research Project\gan_runs\out_wgan"     # any folder outside DATA_PATH
os.makedirs(f"{OUT_DIR}/real", exist_ok=True)
os.makedirs(f"{OUT_DIR}/fake", exist_ok=True)

# ---------- hyper-params -------------
IMG       = 128
BATCH     = 8
EPOCHS    = 1000
Z_DIM     = 128
NGF, NDF  = 64, 64
LR        = 2e-4
BETA1, BETA2 = 0.0, 0.9
GP_LAMBDA = 10.0
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- dataset ------------------
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.Resize((IMG + 16, IMG + 16)),
    transforms.RandomCrop(IMG),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
dataset = datasets.ImageFolder(DATA_PATH, transform)
loader  = DataLoader(dataset, BATCH, shuffle=True, num_workers=2)

# ---------- blocks -------------------
class SelfAttn(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.q = nn.Conv2d(ch, ch // 8, 1)
        self.k = nn.Conv2d(ch, ch // 8, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).view(B, -1, H * W)
        k = self.k(x).view(B, -1, H * W)
        v = self.v(x).view(B, -1, H * W)
        attn = torch.softmax(q.permute(0, 2, 1) @ k, -1)
        out  = (v @ attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x

# ---------- generator ----------------
class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, NGF * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 16), nn.ReLU(True),

            nn.ConvTranspose2d(NGF * 16, NGF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 8),  nn.ReLU(True),
            SelfAttn(NGF * 8),

            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),  nn.ReLU(True),

            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),  nn.ReLU(True),
            SelfAttn(NGF * 2),

            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),     nn.ReLU(True),

            nn.ConvTranspose2d(NGF, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# ---------- discriminator ------------
class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(NDF * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttn(NDF * 2),

            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(NDF * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(NDF * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttn(NDF * 8),

            nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.net(x).view(-1)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0, 0.02)

# ---------- build models -------------
netG, netD = G().to(device), D().to(device)
netG.apply(weights_init); netD.apply(weights_init)

optG = torch.optim.Adam(netG.parameters(), LR, (BETA1, BETA2))
optD = torch.optim.Adam(netD.parameters(), LR, (BETA1, BETA2))
schedG = torch.optim.lr_scheduler.StepLR(optG, 200, 0.5)
schedD = torch.optim.lr_scheduler.StepLR(optD, 200, 0.5)

# ---------- gradient penalty ----------
def grad_penalty(real, fake):
    eps = torch.rand(real.size(0), 1, 1, 1, device=device)
    interp = eps * real + (1 - eps) * fake
    interp.requires_grad_()
    out = netD(interp)
    grad = torch.autograd.grad(out, interp, torch.ones_like(out),
                               create_graph=True, retain_graph=True)[0]
    return ((grad.view(grad.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

# ---------- training loop -------------
fixed = torch.randn(16, Z_DIM, 1, 1, device=device)
for epoch in range(1, EPOCHS + 1):
    for real, _ in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        real = real.to(device)
        b = real.size(0)

        # ---- critic ----
        for _ in range(5):
            z = torch.randn(b, Z_DIM, 1, 1, device=device)
            fake = netG(z).detach()
            lossD = netD(fake).mean() - netD(real).mean() + GP_LAMBDA * grad_penalty(real, fake)
            optD.zero_grad(); lossD.backward(); optD.step()

        # ---- generator ---
        z = torch.randn(b, Z_DIM, 1, 1, device=device)
        fake = netG(z)
        lossG = -netD(fake).mean()
        optG.zero_grad(); lossG.backward(); optG.step()

    schedG.step(); schedD.step()

    if epoch % 25 == 0 or epoch == EPOCHS:
        vutils.save_image((real + 1) / 2,  f"{OUT_DIR}/real/ep{epoch}.png", nrow=4)
        vutils.save_image((fake + 1) / 2,  f"{OUT_DIR}/fake/ep{epoch}.png", nrow=4)
        vutils.save_image((netG(fixed) + 1) / 2, f"{OUT_DIR}/fake/grid_ep{epoch}.png", nrow=4)
        print(f"Saved samples for epoch {epoch}")

torch.save(netG.state_dict(), f"{OUT_DIR}/G_final.pth")
torch.save(netD.state_dict(), f"{OUT_DIR}/D_final.pth")
print("✅ Training done!")

# ---------- evaluation ----------------
def evaluate(real_dir, fake_dir):
    print("🧪 Evaluating …")
    real = sorted([os.path.join(real_dir,f) for f in os.listdir(real_dir) if f.endswith('.png')])[-10:]
    fake = sorted([os.path.join(fake_dir,f) for f in os.listdir(fake_dir) if f.endswith('.png') and 'grid' not in f])[-10:]
    s = []
    for r, f in zip(real, fake):
        r = np.array(Image.open(r).convert('L').resize((IMG, IMG))) / 255.
        f = np.array(Image.open(f).convert('L').resize((IMG, IMG))) / 255.
        s.append(ssim(r, f, data_range=1.0))
    print("Avg SSIM:", np.mean(s))
    fid = fid_score.calculate_fid_given_paths([real_dir, fake_dir], batch_size=8, device=device, dims=2048)
    print("FID:", fid)

evaluate(f"{OUT_DIR}/real", f"{OUT_DIR}/fake")

