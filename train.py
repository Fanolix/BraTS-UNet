import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from tqdm import tqdm

from data import BraTSDataset
from model import UNet

# Config
DATASET_PATH = "c:/Recherche/BraTS/data"
MODEL_PATH = "brats_unet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement fichiers HDF5
h5_files = [os.path.join(DATASET_PATH, f) for f in os.listdir(DATASET_PATH) if f.endswith(".h5")]
h5_files = h5_files[:500]

# Dataset + Split
dataset = BraTSDataset(h5_files)
train_size = int(0.8 * len(dataset))
train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# Modèle
model = UNet().to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.01, 2.0, 2.0]).to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Entraînement
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print("✅ Modèle pré-entraîné chargé.")
else:
    for epoch in range(3):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/3")
        for images, masks in loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / len(train_loader))
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Modèle sauvegardé : {MODEL_PATH}")

# Évaluation
model.eval()
with torch.no_grad():
    images, masks = zip(*[test_dataset[i] for i in range(4)])
    images = torch.stack(images).to(DEVICE)
    masks = torch.stack(masks).to(DEVICE)
    preds = torch.argmax(torch.softmax(model(images), dim=1), dim=1).cpu().numpy()

    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for i in range(4):
        axes[0, i].imshow(images[i][0].cpu(), cmap='gray')
        axes[1, i].imshow(masks[i].cpu(), cmap='jet')
        axes[2, i].imshow(preds[i], cmap='jet')
        axes[0, i].set_title("Scan")
        axes[1, i].set_title("Masque réel")
        axes[2, i].set_title("Prédiction")
        for ax in axes[:, i]:
            ax.axis('off')
    plt.suptitle("Résultat du modèle U-Net")
    plt.tight_layout()
    plt.show()
