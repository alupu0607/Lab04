import os
from datetime import datetime
from torchmetrics import MeanAbsoluteError
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
def extract_date_from_filename(filename):
    date_str = filename.split('_')[2] + filename.split('_')[3]
    return datetime.strptime(date_str, "%Y%m")


def month_difference(start_date, end_date):
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

class CustomImageDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        self.img_pairs = []
        self.transform = transform
        self._load_data(main_dir)

    def _load_data(self, main_dir):
        images_by_label = {}

        for root, dirs, files in os.walk(main_dir):
            if 'images' in dirs:
                img_dir = os.path.join(root, 'images')
                label = os.path.basename(root)

                for file in os.listdir(img_dir):
                    if file.endswith('.tif'):
                        img_path = os.path.join(img_dir, file)
                        rindex = img_path.find('images')
                        date = extract_date_from_filename(img_path[rindex + 7:])
                        if label not in images_by_label:
                            images_by_label[label] = []
                        images_by_label[label].append((img_path, date))

        for label, images in images_by_label.items():
            for i in range(len(images)):
                start_image = images[i]
                for j in range(len(images)):
                    end_image = images[j]
                    time_skip = month_difference(start_image[1], end_image[1])
                    if time_skip > 0:
                        self.img_pairs.append((start_image[0], end_image[0], time_skip, label))
    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, idx):
        start_img_path, end_img_path, time_skip, label = self.img_pairs[idx]

        start_image = Image.open(start_img_path).convert("RGB")
        end_image = Image.open(end_img_path).convert("RGB")
        if self.transform:
            start_image = self.transform(start_image)
            end_image = self.transform(end_image)

        start_image = transforms.ToTensor()(start_image)
        end_image = transforms.ToTensor()(end_image)
        return start_image, end_image, time_skip


class ImageGenerationModel(nn.Module):
    def __init__(self):
        super(ImageGenerationModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.conv2(x)
        return x


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device = 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.mae_metric = MeanAbsoluteError().to(device)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def run(self, n_epochs):
        for epoch in range(n_epochs):
            train_loss, train_accuracy = self.train()
            val_loss, val_accuracy = self.val()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            print(f"Epoch {epoch + 1}/{n_epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train MAE: {train_accuracy:.4f}, "
                  f"Validation Loss: {val_loss:.4f}, Validation MAE: {val_accuracy:.4f}")

        self.plot_metrics()

    def train(self):
        self.model.train()
        total_loss = 0
        total_mae = 0

        for start_image, end_image, time_skip in self.train_loader:
            start_image, end_image = start_image.to(self.device), end_image.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(start_image)
            loss = self.criterion(outputs, end_image)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_mae += self.mae_metric(outputs, end_image).item()

        average_loss = total_loss / len(self.train_loader)
        average_mae = total_mae / len(self.train_loader)
        return average_loss, average_mae

    def val(self):
        self.model.eval()
        total_loss = 0
        total_mae = 0

        with torch.no_grad():
            for start_image, end_image, time_skip in self.val_loader:
                start_image, end_image = start_image.to(self.device), end_image.to(self.device)

                outputs = self.model(start_image)
                loss = self.criterion(outputs, end_image)

                total_loss += loss.item()
                total_mae += self.mae_metric(outputs, end_image).item()

        average_loss = total_loss / len(self.val_loader)
        average_mae = total_mae / len(self.val_loader)
        return average_loss, average_mae

    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Training Accuracy')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        plt.tight_layout()

        plt.savefig('training_validation_metrics.png')
        plt.show()

if __name__ == "__main__":
    main_dir = r"D:\Masters\1st_year\advanced_neural_networks\Advanced-Topics-in-Neural-Networks-Template-2024\Lab04\Dataset"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2,
                               contrast=0.2,
                               saturation=0.2)
        ])
    dataset = CustomImageDataset(main_dir, transform)

    # for the whole dataset
    # dataloader = DataLoader(dataset)
    # for batch in dataloader:
    #     print(batch)
    #     break

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

    model = ImageGenerationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, device= 'cpu')
    trainer.run(n_epochs=10)