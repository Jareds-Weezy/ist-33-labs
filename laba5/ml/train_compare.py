"""Скрипт для сравнения двух архитектур CNN."""

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
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from model.classifier import CIFAR10CNN
from model.classifier_v2 import CIFAR10CNNv2, count_parameters

# Гиперпараметры (одинаковые для обеих моделей)
BATCH_SIZE = 128
NUM_EPOCHS = 20  # Увеличим до 20 для лучшего сравнения
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DATA_DIR = "./ml/data"
WEIGHTS_DIR = Path("./ml/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def get_dataloaders():
    """Подготавливает обучающую и тестовую выборки CIFAR-10."""
    
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


def train_model(model, model_name, train_loader, test_loader, device):
    """Обучает модель и возвращает историю обучения."""
    print(f"\n{'='*60}")
    print(f"Обучение модели: {model_name}")
    print(f"Количество параметров: {count_parameters(model):,}")
    print(f"{'='*60}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    
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
            # Сохраняем лучшие веса
            torch.save(model.state_dict(), WEIGHTS_DIR / f"{model_name}_best.pth")
        
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} | "
              f"train_acc={train_acc*100:.2f}% | "
              f"val_loss={val_loss:.4f} | "
              f"val_acc={val_acc*100:.2f}%")
    
    print(f"\nЛучшая точность на валидации: {history['best_val_acc']*100:.2f}%")
    
    return history


def plot_comparison(histories, model_names):
    """Строит графики сравнения двух моделей."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, NUM_EPOCHS + 1)
    
    # График 1: Train Loss
    ax = axes[0, 0]
    for name, history in histories.items():
        ax.plot(epochs, history['train_loss'], label=f'{name} (train)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Сравнение Train Loss')
    ax.legend()
    ax.grid(True)
    
    # График 2: Val Loss
    ax = axes[0, 1]
    for name, history in histories.items():
        ax.plot(epochs, history['val_loss'], label=f'{name} (val)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Сравнение Validation Loss')
    ax.legend()
    ax.grid(True)
    
    # График 3: Train Accuracy
    ax = axes[1, 0]
    for name, history in histories.items():
        ax.plot(epochs, [acc * 100 for acc in history['train_acc']], label=f'{name}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Сравнение Train Accuracy')
    ax.legend()
    ax.grid(True)
    
    # График 4: Val Accuracy
    ax = axes[1, 1]
    for name, history in histories.items():
        ax.plot(epochs, [acc * 100 for acc in history['val_acc']], label=f'{name}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Сравнение Validation Accuracy')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    # Сохраняем график
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(WEIGHTS_DIR / f"comparison_{timestamp}.png", dpi=150)
    plt.show()
    
    print(f"\nГрафик сохранён в: {WEIGHTS_DIR / f'comparison_{timestamp}.png'}")


def print_comparison_table(histories, model_names):
    """Выводит таблицу сравнения результатов."""
    print("\n" + "="*70)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*70)
    
    # Заголовок таблицы
    print(f"{'Метрика':<25}", end="")
    for name in model_names:
        print(f"{name:<22}", end="")
    print()
    print("-"*70)
    
    # Лучшая точность
    print(f"{'Лучшая val accuracy':<25}", end="")
    for name in model_names:
        best_acc = histories[name]['best_val_acc'] * 100
        print(f"{best_acc:.2f}%{'':<18}", end="")
    print()
    
    # Финальная точность
    print(f"{'Финальная val accuracy':<25}", end="")
    for name in model_names:
        final_acc = histories[name]['val_acc'][-1] * 100
        print(f"{final_acc:.2f}%{'':<18}", end="")
    print()
    
    # Финальный loss
    print(f"{'Финальный val loss':<25}", end="")
    for name in model_names:
        final_loss = histories[name]['val_loss'][-1]
        print(f"{final_loss:.4f}{'':<18}", end="")
    print()
    
    # Количество параметров
    print(f"{'Количество параметров':<25}", end="")
    for name in model_names:
        # Создаём модель для подсчёта параметров
        if 'v1' in name.lower():
            model = CIFAR10CNN()
        else:
            model = CIFAR10CNNv2()
        params = count_parameters(model)
        print(f"{params:,}{'':<15}", end="")
    print()
    print("="*70)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Используется устройство: {device}")
    
    # Загрузка данных
    print("\n[1] Загрузка данных CIFAR-10...")
    train_loader, test_loader = get_dataloaders()
    print(f"    Обучающая выборка: {len(train_loader.dataset)} изображений")
    print(f"    Тестовая выборка: {len(test_loader.dataset)} изображений")
    
    # Создание моделей
    print("\n[2] Создание моделей...")
    model_v1 = CIFAR10CNN().to(device)
    model_v2 = CIFAR10CNNv2().to(device)
    
    print(f"    CIFAR10CNN (v1): {count_parameters(model_v1):,} параметров")
    print(f"    CIFAR10CNNv2 (v2): {count_parameters(model_v2):,} параметров")
    
    if count_parameters(model_v2) > 1_000_000:
        print(f"    ⚠️ ВНИМАНИЕ: Модель v2 превышает лимит 1 млн параметров!")
    
    # Обучение моделей
    print("\n[3] Начало обучения...")
    histories = {}
    
    # Обучение v1
    histories['CIFAR10CNN (v1)'] = train_model(
        model_v1, "cifar10cnn_v1", train_loader, test_loader, device
    )
    
    # Обучение v2
    histories['CIFAR10CNNv2 (v2)'] = train_model(
        model_v2, "cifar10cnn_v2", train_loader, test_loader, device
    )
    
    # Сравнение результатов
    print("\n[4] Сравнение результатов...")
    print_comparison_table(histories, list(histories.keys()))
    
    # Построение графиков
    print("\n[5] Построение графиков...")
    plot_comparison(histories, list(histories.keys()))
    
    print(f"\n[OK] Эксперимент завершён!")
    print(f"    Веса сохранены в: {WEIGHTS_DIR}/")


if __name__ == "__main__":
    main()