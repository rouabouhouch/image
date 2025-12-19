import numpy as np
import cv2
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from VideoSkeleton import VideoSkeleton
from GenVanillaNN import VideoSkeletonDataset, SkeToImageTransform

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# Residual Block pour le générateur
# ============================================================
class ResidualBlock(nn.Module):
    """
    Bloc résiduel simple : Conv -> BN -> ReLU -> Conv -> BN -> Add(input) -> ReLU
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(x + self.block(x))

# ============================================================
# Generator : Skeleton image -> RGB image
# ============================================================
class GenSkeImToImage(nn.Module):
    """
    Generator amélioré : U-Net-ish + ResidualBlock + BatchNorm
    """
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        # Encodeur
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, 2, 1),
            nn.BatchNorm2d(base),
            nn.ReLU(True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base*2, 4, 2, 1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base*2, base*4, 4, 2, 1),
            nn.BatchNorm2d(base*4),
            nn.ReLU(True)
        )

        # Bloc résiduel
        self.res_block = ResidualBlock(base*4)

        # Décodeur
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base*4, base*2, 4, 2, 1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base*2, base, 4, 2, 1),
            nn.BatchNorm2d(base),
            nn.ReLU(True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base, 3, 4, 2, 1),
            nn.Tanh()  # sortie [-1,1]
        )

    def forward(self, x):
        z1 = self.enc1(x)
        z2 = self.enc2(z1)
        z3 = self.enc3(z2)
        z3 = self.res_block(z3)
        z = self.dec3(z3)
        z = self.dec2(z)
        out = self.dec1(z)
        return out

# ============================================================
# Conditional Discriminator : (image + skeleton_image)
# NO BatchNorm (WGAN-GP compliant)
# ============================================================
class Discriminator(nn.Module):
    """
    D(image, skeleton_image) -> score
    """
    def __init__(self, in_channels=6, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base, base*2, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base*2, base*4, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(base*4*8*8, 1)
        )

    def forward(self, img, ske_img):
        x = torch.cat([img, ske_img], dim=1)
        return self.net(x)

# ============================================================
# GAN wrapper (Conditional WGAN-GP + L1)
# ============================================================
class GenGAN:
    def __init__(self, videoSke, loadFromFile=False, batch_size=32):
        image_size = 64

        self.netG = GenSkeImToImage().to(device)
        self.netD = Discriminator().to(device)

        self.lambda_gp = 10.0
        self.lambda_l1 = 50.0
        self.filename = "../data/Dance/DanceGenGAN_Cond.pth"

        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: loading", self.filename)
            self.netG = torch.load(self.filename, map_location='cpu')


            self.netG.eval()

        src_transform = transforms.Compose([
            SkeToImageTransform(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])
        tgt_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])

        self.dataset = VideoSkeletonDataset(
            videoSke,
            ske_reduced=True,
            source_transform=src_transform,
            target_transform=tgt_transform
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )

    def gradient_penalty(self, real, fake, ske):
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)

        d_interpolated = self.netD(interpolated, ske)
        grad_outputs = torch.ones_like(d_interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return penalty

    def train(self, epochs=200):
        optD = torch.optim.Adam(self.netD.parameters(), lr=2e-4, betas=(0.0, 0.9))
        optG = torch.optim.Adam(self.netG.parameters(), lr=2e-4, betas=(0.0, 0.9))
        l1 = nn.L1Loss()

        for e in range(epochs):
            for ske_img, real_img in self.loader:
                ske_img = ske_img.to(device)
                real_img = real_img.to(device)

                # Train Discriminator
                fake_img = self.netG(ske_img).detach()
                d_real = self.netD(real_img, ske_img)
                d_fake = self.netD(fake_img, ske_img)
                gp = self.gradient_penalty(real_img, fake_img, ske_img)
                lossD = -(d_real.mean() - d_fake.mean()) + self.lambda_gp * gp
                optD.zero_grad()
                lossD.backward()
                optD.step()

                # Train Generator
                fake_img = self.netG(ske_img)
                adv = -self.netD(fake_img, ske_img).mean()
                rec = l1(fake_img, real_img)
                lossG = adv + self.lambda_l1 * rec
                optG.zero_grad()
                lossG.backward()
                optG.step()

            print(f"Epoch {e+1} | D: {lossD.item():.3f} | G: {lossG.item():.3f}")

        torch.save(self.netG, self.filename)
        print("Model saved.")

    def generate(self, ske):
        self.netG.eval()
        ske_t = self.dataset.preprocessSkeleton(ske).unsqueeze(0).to(device)
        with torch.no_grad():
            img = self.netG(ske_t)
        img = (img + 1) / 2
        return self.dataset.tensor2image(img[0])

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "../data/taichi1.mp4"
    video = VideoSkeleton(filename)

    gan = GenGAN(video, batch_size=32)
    gan.train(200)

    for i in range(video.skeCount()):
        img = gan.generate(video.ske[i])
        img = cv2.resize(img, (256, 256))
        cv2.imshow("Result", img)
        cv2.waitKey(0)
