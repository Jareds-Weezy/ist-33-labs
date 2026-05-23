"""Обучение CNN на Fashion-MNIST (10 классов одежды)."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from model.classifier_fashion import FashionMNISTCNN

# Гиперпараметры
BATCH_SIZE = 128
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DATA_DIR = "./ml/data"
WEIGHTS_DIR = Path("./ml/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Классы Fashion-MNIST
FASHION_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


def get_dataloaders():
    """Подготавливает обучающую и тестовую выборки Fashion-MNIST."""
    
    # Fashion-MNIST нормализация (среднее и std для этого датасета)
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean=(0.2860,), std=(0.3530,))  # Статистики для Fashion-MNIST
    ])
    
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.2860,), std=(0.3530,))
    ])
    
    # Загрузка Fashion-MNIST
    train_set = torchvision.datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Обучает модель одну эпоху."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return running_loss / len(loader.dataset), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
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


def show_sample_predictions(model, loader, device, num_samples=5):
    """Показывает примеры предсказаний."""
    model.eval()
    
    # Получаем батч
    images, labels = next(iter(loader))
    images, labels = images[:num_samples].to(device), labels[:num_samples]
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)
    
    # Визуализация
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
    for i in range(num_samples):
        img = images[i].cpu().squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {FASHION_CLASSES[labels[i]]}\nPred: {FASHION_CLASSES[predicted[i]]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(WEIGHTS_DIR / "fashion_samples.png", dpi=150)
    plt.show()
    print(f"💾 Примеры предсказаний сохранены: {WEIGHTS_DIR / 'fashion_samples.png'}")


def main():
    print("=" * 60)
    print("ЗАДАНИЕ 4: КЛАССИФИКАЦИЯ FASHION-MNIST (одежда)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Устройство: {device}")
    
    # Загрузка данных
    print("\n📦 Загрузка данных Fashion-MNIST...")
    train_loader, test_loader = get_dataloaders()
    print(f"   Обучающая выборка: {len(train_loader.dataset):,} изображений")
    print(f"   Тестовая выборка: {len(test_loader.dataset):,} изображений")
    print(f"   Количество классов: 10")
    print(f"   Классы: {', '.join(FASHION_CLASSES)}")
    
    # Создание модели
    print("\n🏗️ Создание модели CNN для Fashion-MNIST...")
    model = FashionMNISTCNN(num_classes=10).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Всего параметров: {total_params:,}")
    print(f"   Обучаемых параметров: {trainable_params:,}")
    
    # Обучение
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    
    print("\n🚀 Начало обучения...")
    print("=" * 60)
    
    best_acc = 0.0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_DIR / "fashion_mnist_cnn_best.pth")
            print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | ⭐ НОВАЯ ЛУЧШАЯ!")
        else:
            print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    
    print("\n" + "=" * 60)
    print(f"✅ Обучение завершено!")
    print(f"🏆 Лучшая точность на тесте: {best_acc*100:.2f}%")
    print(f"💾 Веса сохранены в: {WEIGHTS_DIR / 'fashion_mnist_cnn_best.pth'}")
    
    # Построение графиков
    print("\n📈 Построение графиков...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # График точности
    axes[0].plot(history['train_acc'], label='Train Accuracy', marker='o', linewidth=2)
    axes[0].plot(history['val_acc'], label='Test Accuracy', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'Fashion-MNIST: Training Progress (Best: {best_acc*100:.2f}%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # График потерь
    axes[1].plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[1].plot(history['val_loss'], label='Test Loss', marker='s', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Fashion-MNIST: Loss Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(WEIGHTS_DIR / "fashion_mnist_training.png", dpi=150)
    plt.show()
    
    print(f"💾 График сохранён: {WEIGHTS_DIR / 'fashion_mnist_training.png'}")
    
    # Показать примеры предсказаний
    print("\n📸 Примеры предсказаний на тестовых изображениях:")
    show_sample_predictions(model, test_loader, device)
    
    return best_acc


if __name__ == "__main__":
    main()