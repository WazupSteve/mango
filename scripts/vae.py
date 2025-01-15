import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from torchvision.utils import save_image
import numpy as np

# --- Configuration ---
LATENT_DIM = 256
BATCH_SIZE = 64
NUM_EPOCHS = 50  # converges better
IMAGE_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'models/anime_vae_best.pth'
DATA_DIR = 'anime_faces/images'
SAMPLES_DIR = 'samples'
GENERATED_IMAGE_PATH = 'generated_anime_faces_best.png'
LEARNING_RATE = 1e-4 
BETA = 1.0 #stronger KL divergence ( can vary outputs !)
SCHEDULER_STEP_SIZE = 15
SCHEDULER_GAMMA = 0.7

# --- Reproducibility ---
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True

# --- Dataset ---
class AnimeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# --- Encoder ---
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8x8
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, 4, stride=2, padding=1), # 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512)
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# --- Decoder ---
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), # 8x8
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 16x16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),   # 64x64
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)
        return self.decoder(x)

# --- VAE Model ---
class AnimeVAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def generate(self, num_samples, device):
        with torch.no_grad():
            z = torch.randn(num_samples, LATENT_DIM).to(device)
            samples = self.decoder(z)
        return samples

    def save_model(self, path):
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load_model(cls, path, device):
        model = cls(LATENT_DIM)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        return model

# --- Training ---
class VAETrainer:
    def __init__(self, model, device, learning_rate, beta):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.beta = beta
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA) # Added scheduler

    def compute_loss(self, recon_x, x, mu, logvar):
        recon_loss = nn.MSELoss()(recon_x, x)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_div = kl_div / x.numel()  #Average over batch and spatial dimensions
        return recon_loss + self.beta * kl_div

    #training loop
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.compute_loss(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}')
        self.scheduler.step()
        return total_loss / len(train_loader)

# --- Main Function ---
def main():
    #Create directories
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    #Transforms and DataLoader
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # Increased rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),  # Adjusted color jitter
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    dataset = AnimeDataset(DATA_DIR, transform=transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        model = AnimeVAE.load_model(MODEL_PATH, DEVICE)
    else:
        print("Training new model...")
        model = AnimeVAE(LATENT_DIM).to(DEVICE)
        trainer = VAETrainer(model, DEVICE, LEARNING_RATE, BETA)

        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            avg_loss = trainer.train_epoch(train_loader)
            print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')

            #generate and save samples
            with torch.no_grad():
                model.eval()
                samples = model.generate(16, DEVICE)
                save_image(samples, os.path.join(SAMPLES_DIR, f'epoch_{epoch+1}.png'), nrow=4, normalize=True)

        model.save_model(MODEL_PATH)

    print("Generating final samples...")
    model.eval()
    diverse_samples = model.generate(64, DEVICE)
    save_image(diverse_samples, GENERATED_IMAGE_PATH, nrow=8, normalize=True)

if __name__ == '__main__':
    main()