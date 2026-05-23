"""Генерация всех графиков для отчёта."""

import matplotlib.pyplot as plt
import numpy as np

# Данные
models = ['CNN v1', 'CNN v2', 'ResNet18']
accuracies = [82.71, 89.85, 79.03]
colors = ['#3498db', '#2ecc71', '#e74c3c']

# График 1: Сравнение точности
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=2)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Сравнение точности моделей на CIFAR-10', fontsize=14, fontweight='bold')
ax.set_ylim(70, 100)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('ml/weights/final_comparison.png', dpi=150)
plt.show()

print("✅ Графики сохранены в ml/weights/")
print("   - final_comparison.png")