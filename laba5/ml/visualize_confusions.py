"""Визуализация самых путаемых классов для CIFAR-10."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from model.classifier_v2 import CIFAR10CNNv2, CIFAR10_CLASSES

DATA_DIR = "./ml/data"
WEIGHTS_PATH = Path("./ml/weights/cifar10cnn_v2_best.pth")
WEIGHTS_DIR = Path("./ml/weights")
BATCH_SIZE = 64


def get_misclassified_examples(model, loader, device, class1, class2, num_examples=5):
    """Находит примеры, где класс1 был предсказан как класс2."""
    model.eval()
    misclassified = []
    
    # Проверяем существование классов
    if class1 not in CIFAR10_CLASSES or class2 not in CIFAR10_CLASSES:
        print(f"  ⚠️ Пропускаем пару {class1} → {class2}: классы не в CIFAR-10")
        return []
    
    class1_idx = CIFAR10_CLASSES.index(class1)
    class2_idx = CIFAR10_CLASSES.index(class2)
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            # Находим где истинный класс1, а предсказанный класс2
            mask = (labels == class1_idx) & (predicted == class2_idx)
            if mask.any():
                for i in range(len(images)):
                    if mask[i] and len(misclassified) < num_examples:
                        misclassified.append((images[i].cpu(), labels[i].cpu(), predicted[i].cpu()))
            
            if len(misclassified) >= num_examples:
                break
    
    return misclassified


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Проверка существования весов
    if not WEIGHTS_PATH.exists():
        print(f"❌ Веса не найдены: {WEIGHTS_PATH}")
        return
    
    # Загрузка модели
    print("Загрузка модели...")
    model = CIFAR10CNNv2(num_classes=10).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=False))
    model.eval()
    print("✅ Модель загружена")
    
    # Загрузка данных
    print("Загрузка тестовых данных...")
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
    ])
    
    test_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    print(f"✅ Загружено {len(test_set)} изображений")
    
    # Только классы CIFAR-10 (убираем shirt и t-shirt)
    confused_pairs = [
        ('cat', 'dog'),      # 12% ошибок
        ('cat', 'deer'),     # 3% ошибок
        ('bird', 'airplane'), # 2% ошибок
        ('dog', 'cat'),      # 7% ошибок
    ]
    
    print("\nПоиск примеров ошибочной классификации...")
    
    # Создаём фигуру
    fig, axes = plt.subplots(len(confused_pairs), 5, figsize=(15, 3*len(confused_pairs)))
    
    # Если только одна строка, делаем axes двумерным
    if len(confused_pairs) == 1:
        axes = axes.reshape(1, -1)
    
    for row, (class1, class2) in enumerate(confused_pairs):
        print(f"  Поиск: {class1} → {class2}")
        examples = get_misclassified_examples(model, test_loader, device, class1, class2)
        
        for col in range(5):
            if col < len(examples):
                img, true_label, pred_label = examples[col]
                # Денормализация
                mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
                img = img * std + mean
                img = img.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                
                axes[row, col].imshow(img)
                axes[row, col].set_title(f'True: {class1}\nPred: {class2}', fontsize=10)
                axes[row, col].axis('off')
            else:
                axes[row, col].axis('off')
    
    plt.suptitle('Примеры ошибочной классификации (CIFAR-10)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Сохраняем
    save_path = WEIGHTS_DIR / "misclassified_examples.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n💾 Примеры ошибок сохранены: {save_path}")


if __name__ == "__main__":
    main()