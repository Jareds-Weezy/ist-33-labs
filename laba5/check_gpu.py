"""Проверка доступности GPU для PyTorch."""

import torch

print("=" * 50)
print("ПРОВЕРКА GPU")
print("=" * 50)

# 1. Проверка CUDA
cuda_available = torch.cuda.is_available()
print(f"\n1. CUDA доступна: {cuda_available}")

if cuda_available:
    # 2. Количество GPU
    device_count = torch.cuda.device_count()
    print(f"\n2. Количество GPU: {device_count}")
    
    # 3. Информация о каждом GPU
    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"   GPU {i}: {gpu_name}")
    
    # 4. Версия CUDA
    cuda_version = torch.version.cuda
    print(f"\n3. Версия CUDA (PyTorch): {cuda_version}")
    
    # 5. Текущее устройство
    current_device = torch.cuda.current_device()
    print(f"\n4. Текущее устройство: {current_device}")
    
    # 6. Тест производительности (простая операция)
    print("\n5. Тест GPU (матричное умножение)...")
    x = torch.randn(5000, 5000).cuda()
    y = torch.randn(5000, 5000).cuda()
    z = torch.matmul(x, y)
    print("   ✅ GPU работает нормально!")
    
    # 7. Рекомендация
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТ: GPU готов к использованию! ✅")
    print("=" * 50)
    print("\nЗапуск обучения:")
    print("  python -m ml.train_compare")
    
else:
    print("\n❌ CUDA НЕ доступна!")
    print("\nВозможные причины:")
    print("  1. Установлена CPU-версия PyTorch")
    print("  2. Не установлены драйверы NVIDIA")
    print("  3. Нет физической видеокарты NVIDIA")
    
    print("\nДля установки GPU-версии PyTorch:")
    print("  pip uninstall torch torchvision")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")