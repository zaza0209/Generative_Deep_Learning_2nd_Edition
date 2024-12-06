# vae_fashion_pytorch.ipynb

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from scipy.stats import norm

# Load the Fashion-MNIST dataset
transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, 128, 4, 4)
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar, beta=1):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

# Initialize the model, optimizer, and other parameters
input_dim = 32 * 32
latent_dim = 2
model = VAE(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training loop
def train(epochs):
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            if batch_idx == 2:
                break
            data = data.view(-1, 1, 32, 32)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar, beta=500)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item() / len(data):.6f}')
        
        print(f'====> Epoch: {epoch+1}, Average loss: {train_loss / len(train_loader.dataset):.4f}')

# Train the model
train(epochs=5)

# Save the model
torch.save(model.state_dict(), './models/vae_pytorch.pth')

# Reconstruct images
def reconstruct_images(model, test_loader):
    model.eval()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, input_dim)
            recon_batch, _, _ = model(data)
            break
    
    n = min(data.size(0), 8)
    comparison = torch.cat([data[:n], recon_batch.view(100, 1, 32, 32)[:n]])
    return comparison

# Display reconstructed images
reconstructed_images = reconstruct_images(model, test_loader)
plt.figure(figsize=(12, 6))
plt.axis('off')
plt.title('Original vs Reconstructed')
plt.imshow(np.transpose(make_grid(reconstructed_images, nrow=8).numpy(), (1, 2, 0)))
plt.show()

# Embed images into latent space
def embed_images(model, test_loader):
    model.eval()
    with torch.no_grad():
        z_list = []
        for data, _ in test_loader:
            data = data.view(-1, input_dim)
            mu, _ = model.encode(data)
            z_list.append(mu)
    
    z = torch.cat(z_list, dim=0).numpy()
    return z

# Display embedded points in 2D space
z = embed_images(model, test_loader)
plt.figure(figsize=(8, 8))
plt.scatter(z[:, 0], z[:, 1], c='black', alpha=0.5, s=3)
plt.show()

# Generate new images from latent space
def generate_images(model, n=16):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, latent_dim)
        samples = model.decode(z)
    
    return samples

# Display generated images
generated_images = generate_images(model)
plt.figure(figsize=(8, 8))
plt.axis('off')
plt.title('Generated Images')
plt.imshow(np.transpose(make_grid(generated_images.view(16, 1, 32, 32), nrow=4).numpy(), (1, 2, 0)))
plt.show()