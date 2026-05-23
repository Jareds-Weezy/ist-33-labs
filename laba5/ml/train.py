"""Скрипт обучения CNN на CIFAR-10."""

import sys
from pathlib import Path

# Добавляем корневую папку в путь
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from model.classifier import CIFAR10CNN

# Гиперпараметры
BATCH_SIZE = 128
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DATA_DIR = "./ml/data"
WEIGHTS_PATH = Path("./ml/weights/cifar10_cnn.pth")


def get_dataloaders() -> tuple[DataLoader, DataLoader]:
    """Подготавливает обучающую и тестовую выборки CIFAR-10."""
    
    # Аугментации только для обучающей выборки
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
    ])
    
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
    ])
    
    train_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """Обучает модель одну эпоху."""
    model.train()
    running_loss = 0.0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """Оценивает модель."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Используется устройство: {device}")
    
    train_loader, test_loader = get_dataloaders()
    
    model = CIFAR10CNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_acc = 0.0
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Начало обучения ===\n")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_acc={val_acc * 100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"  -> сохранены веса (acc={best_acc * 100:.2f}%)")
    
    print(f"\n[OK] Обучение завершено. Лучшая точность: {best_acc * 100:.2f}%")
    print(f"[OK] Веса сохранены в: {WEIGHTS_PATH.resolve()}")


if __name__ == "__main__":
    main()