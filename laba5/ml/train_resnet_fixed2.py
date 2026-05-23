"""Полностью исправленная версия ResNet18 с правильной заморозкой."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights

# Гиперпараметры
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

DATA_DIR = "./ml/data"
WEIGHTS_DIR = Path("./ml/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def freeze_layers_correctly(model):
    """Правильная заморозка: оставляем только последний блок и классификатор."""
    
    # 1. Замораживаем ВСЕ параметры
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. Размораживаем только последний residual блок (layer4)
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # 3. Размораживаем классификатор (fc)
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # 4. Также размораживаем BatchNorm слои в layer4 (они нужны для стабильности)
    for module in model.layer4.modules():
        if isinstance(module, nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = True
    
    # Считаем статистику
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   ✅ Заморозка выполнена!")
    print(f"   Всего параметров: {total_params:,}")
    print(f"   Обучаемых: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"   Заморожено: {total_params - trainable_params:,} ({(total_params - trainable_params)/total_params*100:.1f}%)")
    
    return model


def main():
    print("=" * 60)
    print("RESNET18 - ПРАВИЛЬНЫЙ TRANSFER LEARNING")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Устройство: {device}")
    
    # Подготовка данных
    print("\n📦 Загрузка данных CIFAR-10...")
    
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test
    )
    
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"   📊 Обучающая выборка: {len(trainset):,}")
    print(f"   📊 Тестовая выборка: {len(testset):,}")
    
    # Создание модели
    print("\n🏗️ Создание модели ResNet18...")
    
    # Загружаем предобученную модель
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    
    # Адаптируем для изображений 32x32
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    # Заменяем классификатор
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 10)
    )
    
    # Правильная заморозка
    print("\n❄️ Применение правильной заморозки слоёв...")
    model = freeze_layers_correctly(model)
    
    model = model.to(device)
    
    # Обучение только обучаемых параметров
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Обучение
    print("\n🚀 Начало обучения...")
    print("=" * 60)
    
    best_acc = 0.0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(trainset)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(testset)
        val_acc = 100.0 * val_correct / val_total
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step()
        
        # Вывод с индикатором
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        
        status = "⭐ НОВАЯ ЛУЧШАЯ!" if is_best else ""
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Best: {best_acc:.2f}% {status}")
        
        if is_best:
            torch.save(model.state_dict(), WEIGHTS_DIR / "resnet18_cifar10_best.pth")
    
    print("\n" + "=" * 60)
    print(f"✅ Обучение завершено!")
    print(f"🏆 Лучшая точность на валидации: {best_acc:.2f}%")
    print(f"💾 Веса сохранены в: {WEIGHTS_DIR / 'resnet18_cifar10_best.pth'}")
    
    # Построение графиков
    print("\n📈 Построение графиков...")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # График точности
    axes[0].plot(history['train_acc'], label='Train Accuracy', marker='o', linewidth=2)
    axes[0].plot(history['val_acc'], label='Validation Accuracy', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('ResNet18: Training Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # График потерь
    axes[1].plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[1].plot(history['val_loss'], label='Validation Loss', marker='s', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('ResNet18: Loss Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(WEIGHTS_DIR / "resnet18_training_correct.png", dpi=150)
    plt.show()
    
    print(f"💾 График сохранён: {WEIGHTS_DIR / 'resnet18_training_correct.png'}")
    
    return best_acc


if __name__ == "__main__":
    main()