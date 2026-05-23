"""Анализ ошибок модели: матрица ошибок и самые путаемые классы."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

from model.classifier import CIFAR10CNN, CIFAR10_CLASSES
from model.classifier_v2 import CIFAR10CNNv2

# Конфигурация
DATA_DIR = "./ml/data"
WEIGHTS_DIR = Path("./ml/weights")
BATCH_SIZE = 128

# Цветовая схема для матрицы ошибок
CMAP = 'YlOrRd'


def get_test_loader(dataset='cifar10'):
    """Загружает тестовую выборку."""
    if dataset == 'cifar10':
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2470, 0.2435, 0.2616)),
        ])
        test_set = torchvision.datasets.CIFAR10(
            root=DATA_DIR, train=False, download=True, transform=transform
        )
        classes = CIFAR10_CLASSES
    elif dataset == 'fashion':
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.2860,), std=(0.3530,))
        ])
        test_set = torchvision.datasets.FashionMNIST(
            root=DATA_DIR, train=False, download=True, transform=transform
        )
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader, classes


@torch.no_grad()
def get_predictions(model, loader, device):
    """Получает все предсказания и истинные метки."""
    model.eval()
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, classes, title, save_path):
    """Строит и сохраняет матрицу ошибок."""
    plt.figure(figsize=(12, 10))
    
    # Нормализуем матрицу
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Строим heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap=CMAP,
                xticklabels=classes, yticklabels=classes,
                annot_kws={'size': 8})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Истинный класс', fontsize=12)
    plt.xlabel('Предсказанный класс', fontsize=12)
    
    # Поворачиваем метки для лучшей читаемости
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  💾 Матрица ошибок сохранена: {save_path}")


def find_most_confused_pairs(cm, classes, top_k=3):
    """Находит топ-K самых путаемых пар классов."""
    # Исключаем диагональ (правильные предсказания)
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    
    # Находим наибольшие значения
    confused_pairs = []
    for _ in range(top_k):
        max_idx = np.argmax(cm_no_diag)
        row = max_idx // len(classes)
        col = max_idx % len(classes)
        max_value = cm_no_diag[row, col]
        
        confused_pairs.append({
            'true_class': classes[row],
            'pred_class': classes[col],
            'count': max_value,
            'percentage': (max_value / cm[row].sum()) * 100
        })
        
        # Обнуляем найденное значение
        cm_no_diag[row, col] = 0
    
    return confused_pairs


def explain_confusions(confused_pairs):
    """Объясняет причины путаницы классов."""
    explanations = {
        ('cat', 'dog'): "Кошки и собаки имеют схожие формы тела, шерсть и позы",
        ('cat', 'deer'): "Кошки и олени могут иметь похожие силуэты и цвет",
        ('dog', 'deer'): "Собаки и олени могут быть похожи в определённых ракурсах",
        ('bird', 'airplane'): "Птицы и самолёты имеют схожий силуэт с крыльями",
        ('automobile', 'truck'): "Автомобили и грузовики - оба транспортные средства с колёсами",
        ('ship', 'airplane'): "Корабли и самолёты оба имеют длинные формы",
        ('frog', 'cat'): "Лягушки и кошки могут иметь схожие позы и размеры",
        ('horse', 'deer'): "Лошади и олени имеют похожее телосложение",
        ('T-shirt/top', 'Shirt'): "Футболки и рубашки имеют схожий крой",
        ('Pullover', 'Coat'): "Свитера и пальто - оба верхняя одежда",
        ('Sandal', 'Sneaker'): "Сандалии и кроссовки - оба типы обуви",
        ('Bag', 'Coat'): "Сумки могут быть спутаны с элементами одежды",
        ('Trouser', 'Dress'): "Брюки и платья могут иметь схожие нижние части"
    }
    
    for pair in confused_pairs:
        key = (pair['true_class'], pair['pred_class'])
        if key in explanations:
            pair['explanation'] = explanations[key]
        else:
            pair['explanation'] = "Визуальное сходство объектов"


def analyze_model(model_name, model_class, weights_path, dataset='cifar10'):
    """Анализирует модель и возвращает результаты."""
    print(f"\n{'='*60}")
    print(f"📊 Анализ модели: {model_name}")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Устройство: {device}")
    
    # Загрузка данных
    print(f"  Загрузка тестовых данных...")
    test_loader, classes = get_test_loader(dataset)
    print(f"  Тестовая выборка: {len(test_loader.dataset)} изображений")
    
    # Загрузка модели
    print(f"  Загрузка модели из {weights_path.name}...")
    if model_class == CIFAR10CNNv2:
        model = model_class(num_classes=10).to(device)
    else:
        model = model_class(num_classes=10).to(device)
    
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
        model.eval()
        print(f"  ✅ Модель загружена")
    else:
        print(f"  ⚠️ Веса не найдены: {weights_path}")
        return None
    
    # Получение предсказаний
    print(f"  Получение предсказаний...")
    predictions, labels = get_predictions(model, test_loader, device)
    
    # Вычисление точности
    accuracy = (predictions == labels).mean() * 100
    print(f"  ✅ Точность: {accuracy:.2f}%")
    
    # Построение матрицы ошибок
    cm = confusion_matrix(labels, predictions)
    
    # Сохранение матрицы ошибок
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = WEIGHTS_DIR / f"confusion_matrix_{model_name}_{timestamp}.png"
    plot_confusion_matrix(cm, classes, f'Матрица ошибок: {model_name} (Acc: {accuracy:.2f}%)', save_path)
    
    # Нахождение самых путаемых пар
    confused_pairs = find_most_confused_pairs(cm, classes, top_k=5)
    explain_confusions(confused_pairs)
    
    # Вывод самых путаемых классов
    print(f"\n  🔍 ТОП-5 самых путаемых пар классов:")
    print(f"  {'='*55}")
    for i, pair in enumerate(confused_pairs, 1):
        print(f"  {i}. {pair['true_class']} → {pair['pred_class']}")
        print(f"     Количество ошибок: {pair['count']} ({pair['percentage']:.1f}% от класса)")
        print(f"     Причина: {pair['explanation']}")
    
    # Подробный отчёт по классам
    print(f"\n  📋 Детальный отчёт по каждому классу:")
    print(f"  {'='*55}")
    report = classification_report(labels, predictions, target_names=classes, output_dict=True)
    
    class_stats = []
    for i, class_name in enumerate(classes):
        precision = report[class_name]['precision']
        recall = report[class_name]['recall']
        f1 = report[class_name]['f1-score']
        support = report[class_name]['support']
        class_stats.append({
            'class': class_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        })
        print(f"  {class_name:15} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'confused_pairs': confused_pairs,
        'class_stats': class_stats
    }


def plot_comparison_confusion_matrices(results, save_path):
    """Сравнение матриц ошибок разных моделей."""
    if len(results) < 2:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, result in enumerate(results):
        if result is None:
            continue
        
        cm = result['confusion_matrix']
        # Нормализуем
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=False, cmap=CMAP,
                    ax=axes[idx], cbar=True, vmin=0, vmax=1)
        axes[idx].set_title(f"{result['model_name']}\nAccuracy: {result['accuracy']:.2f}%", 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Предсказанный класс')
        axes[idx].set_ylabel('Истинный класс')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\n💾 Сравнение матриц ошибок сохранено: {save_path}")


def main():
    print("=" * 60)
    print("ЗАДАНИЕ 5: АНАЛИЗ ОШИБОК МОДЕЛИ")
    print("=" * 60)
    
    # Анализ лучшей модели (CNNv2) на CIFAR-10
    print("\n🔍 Анализ модели CIFAR10CNNv2 (лучшая модель)")
    results = []
    
    result_v2 = analyze_model(
        model_name="CNNv2_CIFAR10",
        model_class=CIFAR10CNNv2,
        weights_path=WEIGHTS_DIR / "cifar10cnn_v2_best.pth",
        dataset='cifar10'
    )
    results.append(result_v2)
    
    # Дополнительный анализ на Fashion-MNIST (опционально)
    print("\n" + "=" * 60)
    print("🔍 Дополнительный анализ: Fashion-MNIST модель")
    print("=" * 60)
    
    # Для Fashion-MNIST нужно создать временную модель
    from model.classifier_fashion import FashionMNISTCNN
    
    weights_fashion = WEIGHTS_DIR / "fashion_mnist_cnn_best.pth"
    if weights_fashion.exists():
        print(f"  Загрузка Fashion-MNIST модели...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загрузка тестовых данных для Fashion-MNIST
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.2860,), std=(0.3530,))
        ])
        test_set = torchvision.datasets.FashionMNIST(
            root=DATA_DIR, train=False, download=True, transform=transform
        )
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
        
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        # Загрузка модели
        model = FashionMNISTCNN(num_classes=10).to(device)
        model.load_state_dict(torch.load(weights_fashion, map_location=device, weights_only=False))
        model.eval()
        
        # Получение предсказаний
        predictions, labels = get_predictions(model, test_loader, device)
        accuracy = (predictions == labels).mean() * 100
        print(f"  ✅ Точность: {accuracy:.2f}%")
        
        # Построение матрицы ошибок
        cm = confusion_matrix(labels, predictions)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = WEIGHTS_DIR / f"confusion_matrix_FashionMNIST_{timestamp}.png"
        plot_confusion_matrix(cm, classes, f'Fashion-MNIST: Матрица ошибок (Acc: {accuracy:.2f}%)', save_path)
        
        # Нахождение самых путаемых пар
        confused_pairs = find_most_confused_pairs(cm, classes, top_k=5)
        explain_confusions(confused_pairs)
        
        print(f"\n  🔍 ТОП-5 самых путаемых пар классов (Fashion-MNIST):")
        print(f"  {'='*55}")
        for i, pair in enumerate(confused_pairs, 1):
            print(f"  {i}. {pair['true_class']} → {pair['pred_class']}")
            print(f"     Количество ошибок: {pair['count']} ({pair['percentage']:.1f}% от класса)")
            print(f"     Причина: {pair['explanation']}")
    else:
        print(f"  ⚠️ Веса Fashion-MNIST не найдены")
    
    # Вывод итогового анализа
    print("\n" + "=" * 60)
    print("📊 ИТОГОВЫЙ АНАЛИЗ")
    print("=" * 60)
    
    if result_v2:
        print(f"\n🏆 Лучшая модель: {result_v2['model_name']}")
        print(f"🎯 Точность: {result_v2['accuracy']:.2f}%")
        print(f"\n📋 Самые частые ошибки на CIFAR-10:")
        for i, pair in enumerate(result_v2['confused_pairs'][:3], 1):
            print(f"  {i}. {pair['true_class']} ↔ {pair['pred_class']}")
            print(f"     {pair['explanation']}")
    
    print("\n" + "=" * 60)
    print("✅ Анализ ошибок завершён!")
    print("💾 Результаты сохранены в папке ml/weights/")
    print("=" * 60)


if __name__ == "__main__":
    main()