import cv2
import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

import os

IMAGE_SIZE = (64, 64)

images_list = []
labels_list = []

regression_scores = []
blueness_scores = []
lambda_vals = [0, 0.5, 1, 2, 5, 10, 50, 100, 500]
# lambda_vals = [1]

generator_losses = []
discriminator_losses = []

dataroot = "custom_dataset" 
workers = 4
batch_size = 1
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 100
lr = 0.0001
beta1 = 0.5
ngpu = 1         

skip = 1

for file in glob.glob("thumbnails/*.png"):
    img = cv2.imread(file)
    if img is None:
        continue
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img_rgb.shape[0] != IMAGE_SIZE[0] or img_rgb.shape[1] != IMAGE_SIZE[1]:
        img_rgb = cv2.resize(img_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    images_list.append(img_rgb)
    try:
        labels_list.append(int(re.findall(r"(\d+)", file)[0]))
    except Exception as e:
        labels_list.append(0)

initial_labels_list = [i for i in labels_list]
labels_list = [i/(max(labels_list) if max(labels_list) != 0 else 1) for i in labels_list]

images_array = np.array(images_list, dtype=np.uint8)
labels_array = np.array(labels_list, dtype=np.float64)

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor([label], dtype=torch.float32)
            
        return image, label

class SimpleCNNRegression(nn.Module):
    def __init__(self):
        super(SimpleCNNRegression, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        self.full_layer_count = int(32 * 12 * 8 * IMAGE_SIZE[0]/32 * IMAGE_SIZE[1]/48)
        self.fc1 = nn.Linear(self.full_layer_count, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.full_layer_count)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_dataset = CustomImageDataset(images_array, labels_array, transform=transform)

def train_cnn_regression(images, labels, num_epochs=10, batch_size=32, learning_rate=0.001):

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    full_dataset = CustomImageDataset(images, labels, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNNRegression().to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training regression model on {device}...")
    for epoch in range(num_epochs):
        model.train()
        correct, total = 0,0
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, targets = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()

        val_loss = 0.0
        ACC_TOLERANCE = 0.1

        with torch.no_grad():
            for data in val_loader:
                images, targets = data[0].to(device), data[1].to(device)
                outputs = model(images)
                
                for res in abs(outputs - targets):
                    if res <= ACC_TOLERANCE:
                        correct += 1
                    total += 1
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)

        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.3f}, Val Loss: {avg_val_loss:.3f}, Val Accuracy: {val_accuracy:.3f}')

    print('Regression complete!')
    return (model, history)

if __name__ == "__main__":

    trained_reg_model, training_history = train_cnn_regression(
        images=images_array,
        labels=labels_array,
        num_epochs=100,
        batch_size=4
    )

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(

                nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()

            )

        def forward(self, input):
            return self.main(input)


    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(

                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True), 

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)

    if __name__ == '__main__':
        for L in lambda_vals:

            regression_scores = []
            blueness_scores = []
            generator_losses = []
            discriminator_losses = []

            netG = Generator(ngpu).to(device)
            netG.apply(weights_init)

            netD = Discriminator(ngpu).to(device)
            netD.apply(weights_init)

            criterion = nn.BCELoss()

            fixed_noise = torch.randn(64, nz, 1, 1, device=device)

            real_label = 1.
            fake_label = 0.

            optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
            optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

            print(f"Starting Training Loop for regression weight = {L}...")
            for epoch in range(num_epochs):
                for i, data in enumerate(dataloader, 0):
                    
                    netD.zero_grad()
                    real_cpu = data[0].to(device)
                    b_size = real_cpu.size(0)
                    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                    output = netD(real_cpu).view(-1)
                    errD_real = criterion(output, label)
                    errD_real.backward()
                    D_x = output.mean().item()

                    noise = torch.randn(b_size, nz, 1, 1, device=device)
                    fake = netG(noise)
                    label.fill_(fake_label)
                    output = netD(fake.detach()).view(-1)
                    errD_fake = criterion(output, label)
                    errD_fake.backward()
                    D_G_z1 = output.mean().item()
                    errD = errD_real + errD_fake

                    optimizerD.step()

                    netG.zero_grad()
                    label.fill_(real_label)
                    output = netD(fake).view(-1)
                    errG = criterion(output, label)

                    noise = torch.randn(b_size, nz, 1, 1, device=device)
                    fakes = netG(noise).detach().cpu()
                    lambda_blue = L
                    blue_presence = torch.mean(fakes[:, 2, :, :])
                    blue_presence_loss = -blue_presence * lambda_blue

                    trained_reg_model.to(device)

                    regression_lambda = L
                    regr_output = 0

                    with torch.no_grad():
                        fakes = netG(noise).detach().cpu()
                        for fake in fakes:
                            input_tensor = fake.unsqueeze(0).to(device)
                            regr_output += trained_reg_model(input_tensor)[0][0].item()
                            
                        regr_output = regr_output/len(fakes)

                    # errG = errG + regr_output * regression_lambda
                    errG = errG + blue_presence_loss
                        
                    errG.backward()
                    D_G_z2 = output.mean().item()
                    optimizerG.step()

                    if i % 50 == 0:
                        print(f'[{epoch+1}/{num_epochs}][{i+1}/{len(dataloader)}] '
                            f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} ')


                regr_output = 0
                with torch.no_grad():
                    fakes = netG(fixed_noise).detach().cpu()
                    for fake in fakes:
                        input_tensor = fake.unsqueeze(0).to(device)
                        regr_output += trained_reg_model(input_tensor)[0][0].item()
                    regr_output = regr_output/len(fakes)

                generator_losses.append(errG.item())
                discriminator_losses.append(errD.item())
                
                regression_scores.append(regr_output * max(initial_labels_list))
                blueness_scores.append(blue_presence)

                if epoch % skip == 0:

                    try:
                        os.mkdir(f"results/r={L}")
                    except Exception as e:
                        pass

                    vutils.save_image(fake, f'results/r={L}/generated_images_epoch_{epoch+1}.png', normalize=True)

                plt.plot(range(epoch+1), regression_scores)
                plt.xlabel("Epoch num")
                plt.ylabel("Predicted view count")
                plt.title(f"Regression output of each generation of image outputs (L={L})")

                try:
                    os.mkdir(f"charts/r={L}")
                except Exception as e:
                    pass

                plt.savefig(f"charts/r={L}/regression_chart.png")
                plt.clf()

                plt.plot(range(epoch+1), blueness_scores)
                plt.xlabel("Epoch num")
                plt.ylabel("Mean value of blue channel")
                plt.title(f"Mean blueness of each generation of image outputs (L={L})")
                plt.savefig(f"charts/r={L}/blueness_chart.png")
                plt.clf()

                plt.plot(range(epoch+1), generator_losses, label="Generator loss", color = "b")
                plt.plot(range(epoch+1), discriminator_losses, label = "Discriminator loss", color = "r")
                plt.legend()
                plt.xlabel("Epoch num")
                plt.ylabel("Loss")
                plt.title(f"Comparison of generator/discriminator losses over time (L={L})")

                plt.savefig(f"charts/r={L}/losses_chart.png")
                plt.clf()

            print("Training complete. Check results folder for output.")

            list_of_outputs = []

            for i in range(100):
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                with torch.no_grad():
                    fakes = netG(noise).detach().cpu()
                    fake_out = fakes[0]
                    input_tensor = fakes[0].unsqueeze(0).to(device)
                    list_of_outputs.append((fake_out, trained_reg_model(input_tensor)[0][0].item()))
            
            list_of_outputs.sort(key=lambda x: x[1], reverse=True)

            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            axes = axes.flatten()

            for i, winner in enumerate(list_of_outputs[:10]):
                (winning_image, winning_measure) = winner
                image_display = winning_image.numpy().transpose((1, 2, 0))
                # Denormalize for viewing (optional, but makes colors right)
                image_display = image_display * 0.5 + 0.5 
                image_display = np.clip(image_display, 0, 1)

                print(i)

                if (i < len(axes)):
                    axes[i].imshow(image_display)
                    axes[i].set_title(f"Predicted View Count: {(winning_measure * max(initial_labels_list)):.0f}")

            plt.show()