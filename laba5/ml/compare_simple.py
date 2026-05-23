"""Простое сравнение всех трёх моделей."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

from model.classifier import CIFAR10CNN
from model.classifier_v2 import CIFAR10CNNv2
from torchvision.models import resnet18

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


def load_resnet18_from_weights(weights_path, device):
    """Загружает ResNet18 из сохранённых весов."""
    print(f"  Загрузка ResNet18 из {weights_path.name}...")
    
    # Создаём модель с той же архитектурой, что при обучении
    from torchvision.models import resnet18
    
    model = resnet18(pretrained=False)
    
    # Адаптируем для 32x32
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    # Заменяем fc слой на тот же, что использовался при обучении
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 10)
    )
    
    # Загружаем веса
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
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
    
    results = {}
    
    # 1. Модель v1
    print("\n  📦 Загрузка CIFAR10CNN (v1)...")
    weights_v1 = WEIGHTS_DIR / "cifar10cnn_v1_best.pth"
    if weights_v1.exists():
        model_v1 = CIFAR10CNN(num_classes=10).to(device)
        model_v1.load_state_dict(torch.load(weights_v1, map_location='cpu', weights_only=False))
        model_v1.eval()
        acc_v1 = evaluate_model(model_v1, test_loader, device)
        results['CIFAR10CNN (v1)'] = acc_v1
        print(f"    ✅ Точность: {acc_v1*100:.2f}%")
    else:
        print(f"    ⚠️ Веса не найдены")
    
    # 2. Модель v2
    print("\n  📦 Загрузка CIFAR10CNNv2 (v2)...")
    weights_v2 = WEIGHTS_DIR / "cifar10cnn_v2_best.pth"
    if weights_v2.exists():
        model_v2 = CIFAR10CNNv2(num_classes=10).to(device)
        model_v2.load_state_dict(torch.load(weights_v2, map_location='cpu', weights_only=False))
        model_v2.eval()
        acc_v2 = evaluate_model(model_v2, test_loader, device)
        results['CIFAR10CNNv2 (v2)'] = acc_v2
        print(f"    ✅ Точность: {acc_v2*100:.2f}%")
    else:
        print(f"    ⚠️ Веса не найдены")
    
    # 3. ResNet18
    print("\n  📦 Загрузка ResNet18...")
    weights_resnet = WEIGHTS_DIR / "resnet18_cifar10_best.pth"
    if weights_resnet.exists():
        try:
            model_resnet = load_resnet18_from_weights(weights_resnet, device)
            acc_resnet = evaluate_model(model_resnet, test_loader, device)
            results['ResNet18 (Transfer Learning)'] = acc_resnet
            print(f"    ✅ Точность: {acc_resnet*100:.2f}%")
        except Exception as e:
            print(f"    ❌ Ошибка загрузки: {e}")
    else:
        print(f"    ⚠️ Веса не найдены")
    
    # Вывод результатов
    print("\n" + "=" * 60)
    print("[3] РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("=" * 60)
    
    for name, acc in results.items():
        print(f"{name:35} | Accuracy: {acc*100:.2f}%")
    
    print("=" * 60)
    
    # Построение графика
    if results:
        print("\n[4] Построение графика...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = list(results.keys())
        accuracies = [results[name] * 100 for name in names]
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        bars = ax.bar(names, accuracies, color=colors[:len(names)], edgecolor='black', linewidth=1.5)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Сравнение моделей классификации CIFAR-10', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        
        # Сохраняем график
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(WEIGHTS_DIR / f"comparison_final_{timestamp}.png", dpi=150)
        plt.show()
        
        print(f"\n💾 График сохранён: {WEIGHTS_DIR / f'comparison_final_{timestamp}.png'}")
        
        # Заключение
        print("\n" + "=" * 60)
        print("📊 ЗАКЛЮЧЕНИЕ")
        print("=" * 60)
        
        best_model = max(results, key=results.get)
        best_acc = results[best_model] * 100
        
        print(f"\n🏆 Лучшая модель: {best_model}")
        print(f"🎯 Точность: {best_acc:.2f}%")
        
        if 'v2' in best_model:
            print(f"\n✅ Собственная архитектура CNN v2 показала лучший результат!")
        elif 'ResNet' in best_model:
            print(f"\n✅ Transfer Learning с ResNet18 показал лучший результат!")
        
        print("=" * 60)


if __name__ == "__main__":
    main()