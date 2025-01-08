import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils import MyLogger
from gradcam import GradCAM

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ResNet9(nn.Module):
    def __init__(self, num_classes=10, num_color_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_color_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        self.resblock1 = ResidualBlock(128, 128)
        self.resblock2 = ResidualBlock(128, 256, stride=2)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.resblock3 = ResidualBlock(512, 512)

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.resblock3(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))

    def num_parameters(self):
        print(f"Total Params: {sum(p.numel() for p in self.parameters())} | Trainable Params: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze(self, *layers):
        if len(layers) == 0:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for layer in layers:
                for param in self.__getattr__(layer).parameters():
                    param.requires_grad = False
    
    def unfreeze(self, *layers):
        for layer in layers:
            for param in self.__getattr__(layer).parameters():
                param.requires_grad = True
    
def start_run(model, train_loader, val_loader, test_loader, criterion, optimizer, config, target_layer="resblock1.conv1"):
    device = torch.device(config.train["device"])
    model = model.to(device)
    
    num_epochs = config.train["num_epochs"]
    grad_clip = config.train["grad_clip"]
    grad_accumulation_steps = config.train["grad_accum_steps"]
    do_validation = config.train["do_validation"]

    print('Starting Training with the following configuration:')
    print(config)

    if config.logging['print_every'] != 0:
        logger = MyLogger()
    
    for epoch in range(num_epochs):
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        optimizer.zero_grad()
        for step, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch+1}/{num_epochs}", leave=False), 1):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / grad_accumulation_steps
            
            loss.backward()
            
            if step % grad_accumulation_steps == 0:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        
        logdict = {"epoch": epoch+1, "train_loss": epoch_loss, "train_accuracy": train_accuracy}
        
        if do_validation:
            model.eval()
            val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
            logdict["val_loss"] = val_loss
            logdict["val_accuracy"] = val_accuracy
        
        if (epoch+1) % config.logging['print_every'] == 0:
            logger.log_metrics(logdict)
            gc = GradCAM(model)

            random_batch_idx = random.randint(0, len(val_loader) - 1)
            for idx, (images, labels) in enumerate(val_loader):
                if idx == random_batch_idx:
                    random_img_idx = random.randint(0, len(images) - 1)
                    gc.visualize(images[random_img_idx], target_layer, alpha=0.5, target_class=labels[random_img_idx].item())
                    break

    if config.logging["save_model"]:
        model_save_path = config.logging["model_save_path"]
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")
    
    print(f"Test Accuracy: {evaluate_model(model, test_loader, criterion, device)[1]:.2%}")
    
    return model, logger

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    model = model.to(device)
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    loss = running_loss / len(dataloader)
    accuracy = correct / total
    return loss, accuracy

def set_global_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)