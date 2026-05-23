"""Сравнение всех трёх моделей: v1, v2 и ResNet18."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from model.classifier import CIFAR10CNN
from model.classifier_v2 import CIFAR10CNNv2
from model.resnet_cifar10 import ResNet18CIFAR10

BATCH_SIZE = 128
DATA_DIR = "./ml/data"
WEIGHTS_DIR = Path("./ml/weights")


def get_test_loader():
    """Загружает тестовую выборку."""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
    ])
    
    test_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    
    return DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Оценивает модель на тестовой выборке."""
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return correct / total


def load_model(model_class, weights_path, num_classes=10):
    """Загружает модель из весов."""
    model = model_class(num_classes=num_classes)
    
    if weights_path.exists():
        # Добавляем weights_only=False для совместимости
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
        print(f"  ✅ Загружены веса: {weights_path.name}")
    else:
        print(f"  ⚠️ Веса не найдены: {weights_path.name}")
    
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Используется устройство: {device}")
    
    # Загрузка тестовых данных
    print("\n[1] Загрузка тестовой выборки...")
    test_loader = get_test_loader()
    print(f"    Тестовая выборка: {len(test_loader.dataset)} изображений")
    
    # Загрузка моделей
    print("\n[2] Загрузка обученных моделей...")
    
    models = {
        'CIFAR10CNN (v1)': (CIFAR10CNN, WEIGHTS_DIR / "cifar10cnn_v1_best.pth"),
        'CIFAR10CNNv2 (v2)': (CIFAR10CNNv2, WEIGHTS_DIR / "cifar10cnn_v2_best.pth"),
        'ResNet18 (Transfer Learning)': (ResNet18CIFAR10, WEIGHTS_DIR / "resnet18_cifar10_best.pth"),
    }
    
    loaded_models = {}
    for name, (model_class, weights_path) in models.items():
        print(f"\n  {name}:")
        model = load_model(model_class, weights_path)
        loaded_models[name] = model.to(device)
    
    # Оценка
    print("\n[3] Оценка моделей на тестовой выборке...")
    print("=" * 60)
    
    results = {}
    for name, model in loaded_models.items():
        accuracy = evaluate_model(model, test_loader, device)
        results[name] = accuracy
        print(f"{name:35} | Accuracy: {accuracy*100:.2f}%")
    
    print("=" * 60)
    
    # Визуализация сравнения
    print("\n[4] Построение графика сравнения...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(results.keys())
    accuracies = [results[name] * 100 for name in names]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars = ax.bar(names, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    # Добавляем значения на столбцы
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Сравнение моделей классификации CIFAR-10', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Поворачиваем подписи
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    # Сохраняем график
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(WEIGHTS_DIR / f"all_models_comparison_{timestamp}.png", dpi=150)
    plt.show()
    
    print(f"\nГрафик сохранён в: {WEIGHTS_DIR / f'all_models_comparison_{timestamp}.png'}")
    
    # Вывод заключения
    print("\n" + "=" * 60)
    print("ЗАКЛЮЧЕНИЕ")
    print("=" * 60)
    
    best_model = max(results, key=results.get)
    best_acc = results[best_model] * 100
    
    print(f"\n📊 Итоговые результаты:")
    for name, acc in results.items():
        print(f"   {name}: {acc*100:.2f}%")
    
    print(f"\n🏆 Лучшая модель: {best_model}")
    print(f"🎯 Точность: {best_acc:.2f}%")
    
    # Анализ
    print("\n📝 Анализ:")
    v2_acc = results['CIFAR10CNNv2 (v2)'] * 100
    v1_acc = results['CIFAR10CNN (v1)'] * 100
    resnet_acc = results['ResNet18 (Transfer Learning)'] * 100
    
    print(f"   • CNN v2 лучше CNN v1 на {v2_acc - v1_acc:.2f}%")
    print(f"   • CNN v2 лучше ResNet18 на {v2_acc - resnet_acc:.2f}%")
    
    if v2_acc > resnet_acc:
        print(f"\n   ✅ Вывод: Собственная архитектура CNN v2 показала лучший результат!")
        print(f"      Преимущества: оптимизирована для маленьких изображений (32x32)")
    else:
        print(f"\n   ✅ Вывод: Transfer Learning с ResNet18 показал лучший результат!")
        print(f"      Преимущества: использование предобученных признаков ImageNet")
    
    print("=" * 60)


if __name__ == "__main__":
    main()