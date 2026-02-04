#1. import required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
import matplotlib.pyplot as plt
#state the version of pytorch
# print(torch.__version__)
# version should be after '2.6.0+cu124'

# print(torchvision.__version__)
# version should be after 0.21.0+cu124 chesk by running

#2. set up device agnostic code
# print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

#3. set the seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)

#4. setting the hyperparameters
BATCH_SIZE = 128 #use multiples of 8
EPOCHS = 10 #Try increasing epochs to 30
LEARNING_RATE = 3e-4 #0.0003
PATCH_SIZE = 4
NUM_CLASSES = 10
IMAGE_SIZE = 32 #transform the image and make the size go to 224
CHANNELS = 3
EMBED_DIM = 256
NUM_HEADS = 8 #increase the number of heads
DEPTH = 6
MLP_DIM = 512 #DIM is dimention MLP is multiplex
DROP_RATE = 0.1

#5. Define image Transformations
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5), (0.5))
#     #1. Helps model to converge faster
#     #2. Helps to make numerical computation stable
# ])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

#6. Getting a dataset
train_dataset = datasets.CIFAR10(root="data",
                                 train=True,
                                 download=True,
                                 transform=transform_train)
#chech documentation for this dataset
test_dataset = datasets.CIFAR10(root="data",
                                train=False,
                                download=True,
                                transform=transform_train)
# print(train_dataset)
# print(test_dataset)

# print(len(train_dataset))
# print(len(test_dataset))

#7. Converting our datasets into dataloaders
#Right now our data is in the form of PyTorch Datasets
#Dataloader turns our data into batches or (mini-batches)
#why do we need this?
# 1 it is more computationally efficient, as in,
# your computing hardware may not be
# able to look (store in memory)at 50000 images in one hit
# so we can break it into 128 images at a time. (batch size of 128).
# 2 it gives our neural network more chances to update its gradient per epoch
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)
# check what is created with print
# print(f"DataLoader: {train_loader, test_loader}")
# print(f"Length of train_loader: {len(train_loader)} batches of {BATCH_SIZE}...")
# print(f"Length of test_loader: {len(test_loader)} batches of {BATCH_SIZE}...")
# 128 images per batch and 391 batches gives 50048 images 391 is rounded true number of batches is 50000/128

#BUILDING VISION TRANSFORMER MODEL WITH COMPONENTS FROM SCRATCH
#remember PATCH_SIZE = 4
class PatchEmbedding(nn.Module):
    def __init__(self,
                 img_size,
                 patch_size,
                 in_channels,
                 embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels=in_channels,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
    
    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.proj(x) #(B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) #(B, N, E)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1) 
        x = x + self.pos_embed
        return x
    
class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features,
                             out_features=hidden_features)# fully connected layer fc
        
        self.fc2 = nn.Linear(in_features=hidden_features,
                             out_features=in_features)
        self.dropout = nn.Dropout(drop_rate)
    def forward(self, x):
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, drop_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, drop_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_dim, drop_rate):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.encoder = nn.Sequential(
            *[TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, drop_rate) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)

#Instantiate a Model
model = VisionTransformer(
    IMAGE_SIZE, PATCH_SIZE, CHANNELS, NUM_CLASSES,
    EMBED_DIM, DEPTH, NUM_HEADS, MLP_DIM, DROP_RATE
).to(device)
print(model.state_dict())

#9. Defining a loss function and an optimizer
criterion = nn.CrossEntropyLoss()# Measures how wrong our model is
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=LEARNING_RATE)# updates our model parameters to try and reduce loss
# print(criterion)
# print(optimizer)

#10. Defining a training loop function
def train(model, loader, optimizer, criterion):
    #set the mode of the model into training
    model.train()
    total_loss, correct = 0, 0
    # x=batch of images y=batch of labels per targer
    for x, y in loader:
        #moving or sending our data into the target device
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        #1. Forward pass (model outputs raw logits)
        out = model(x)
        #2. calculate loss (per batch)
        loss = criterion(out, y)
        #3. perform backpropogation
        loss.backward()
        #4. perform gradient descent
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
    #you have to scan the loss (Normalize step to make the loss general accross all batches)
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate(model, loader):
    model.eval() # set the mode of the model into evaluation
    correct = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(dim=1) == y).sum().item()
    return correct / len(loader.dataset)

# Progress bar library(optional)
# from tqdm.auto import tqdm
# Training
train_accuracies, test_accuracies = [], []
for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    test_acc = evaluate(model, test_loader)
    train_accuracies.append(test_acc)
    test_accuracies.append(test_acc)
    print(f"Epoch: {epoch+1}/{EPOCHS}, Train loss: {train_loss:.4f}, Train acc:{train_acc:.4f}%, Test acc:{test_acc:.4f}%")

# plot accuracy
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and test Accuracy")
plt.show()

import random
def predict_and_plot_grid(model,
                          dataset,
                          classes,
                          grid_size=3):
    model.eval()
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(9, 9))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = random.randint(0, len(dataset) - 1)
            img, true_label = dataset[idx]
            input_tensor = img.unsqueeze(dim=0).to(device)
            with torch.inference_mode():
                output = model(input_tensor)
                _, predicted = torch.max(output.data, 1)
            img = img / 2 + 0.5 #Un normalize our images to be able to plot them with matplotlib as negative pixles are not plotted
            npimg = img.cpu().numpy()
            axes[i, j].imshow(np.transpose(npimg, (1, 2, 0)))
            truth = classes[true_label] == classes[predicted.item()]
            if truth:
                color = "g"
            else:
                color = "r"
            axes[i, j].set_title(f"Truth: {classes[true_label]}\nPredicted: {classes[predicted.item()]}", fontsize=10, c=color)
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.show()

predict_and_plot_grid(model, test_dataset, classes=train_dataset.classes, grid_size=3)