"""Обучение ResNet18 с Transfer Learning для CIFAR-10."""

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

from model.resnet_cifar10 import ResNet18CIFAR10, count_trainable_parameters

# Гиперпараметры
BATCH_SIZE = 128
NUM_EPOCHS = 5  # Для Transfer Learning достаточно 5 эпох
LEARNING_RATE = 1e-3  # Выше чем при полном обучении
WEIGHT_DECAY = 1e-4

DATA_DIR = "./ml/data"
WEIGHTS_DIR = Path("./ml/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def get_dataloaders():
    """Подготавливает данные с аугментацией для Transfer Learning."""
    
    # Более сильная аугментация для Transfer Learning
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),  # Добавляем повороты
        T.ColorJitter(brightness=0.2, contrast=0.2),  # Изменение цвета
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


def train_resnet():
    """Основная функция обучения ResNet18."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Используется устройство: {device}")
    
    # Загрузка данных
    print("\n[1] Загрузка данных CIFAR-10...")
    train_loader, test_loader = get_dataloaders()
    print(f"    Обучающая выборка: {len(train_loader.dataset)} изображений")
    print(f"    Тестовая выборка: {len(test_loader.dataset)} изображений")
    
    # Создание модели
    print("\n[2] Создание модели ResNet18...")
    model = ResNet18CIFAR10(num_classes=10, freeze_backbone=True).to(device)
    
    param_stats = count_trainable_parameters(model)
    print(f"    Общее количество параметров: {param_stats['total']:,}")
    print(f"    Обучаемых параметров: {param_stats['trainable']:,}")
    print(f"    Процент обучаемых: {param_stats['trainable_percent']:.2f}%")
    
    # Optimizer только для обучаемых параметров
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # Обучение
    print("\n[3] Начало обучения (Transfer Learning)...")
    print("=" * 60)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_val_acc': 0.0
    }
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            torch.save(model.state_dict(), WEIGHTS_DIR / "resnet18_cifar10_best.pth")
            print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                  f"train_loss={train_loss:.4f} | "
                  f"train_acc={train_acc*100:.2f}% | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_acc={val_acc*100:.2f}% | "
                  f"⭐ СОХРАНЕНО")
        else:
            print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                  f"train_loss={train_loss:.4f} | "
                  f"train_acc={train_acc*100:.2f}% | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_acc={val_acc*100:.2f}%")
    
    print(f"\nЛучшая точность: {history['best_val_acc']*100:.2f}%")
    
    # Сохранение модели
    torch.save(model.state_dict(), WEIGHTS_DIR / "resnet18_cifar10_final.pth")
    print(f"[OK] Модель сохранена в: {WEIGHTS_DIR}/resnet18_cifar10_final.pth")
    
    return history


def plot_training_history(history):
    """Построение графика обучения."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, NUM_EPOCHS + 1)
    
    # График Loss
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    ax.plot(epochs, history['val_loss'], label='Val Loss', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('ResNet18: Training and Validation Loss')
    ax.legend()
    ax.grid(True)
    
    # График Accuracy
    ax = axes[1]
    ax.plot(epochs, [acc * 100 for acc in history['train_acc']], label='Train Acc', marker='o')
    ax.plot(epochs, [acc * 100 for acc in history['val_acc']], label='Val Acc', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('ResNet18: Training and Validation Accuracy')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    # Сохраняем график
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(WEIGHTS_DIR / f"resnet_training_{timestamp}.png", dpi=150)
    plt.show()
    
    print(f"\nГрафик сохранён в: {WEIGHTS_DIR / f'resnet_training_{timestamp}.png'}")


def main():
    print("=" * 60)
    print("ЗАДАНИЕ 3: Transfer Learning с ResNet18")
    print("=" * 60)
    
    # Обучение
    history = train_resnet()
    
    # Визуализация
    print("\n[4] Построение графиков...")
    plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("✅ Transfer Learning завершён!")
    print(f"Итоговая точность: {history['best_val_acc']*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()