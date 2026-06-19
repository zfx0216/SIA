import matplotlib.pyplot as plt

# Global font configuration
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# X-axis: 1 - SSIM
x = [0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0045, 0.0050]

# Dataset
resnet50 = {
    "PGD": [2, 9.5, 22.8, 34.7, 47.1, 57.0, 66.2, 72.3, 77.4, 82.1],
    "CW": [0.3, 0.9, 5.1, 9.2, 11.5, 17.5, 23.6, 27.4, 32.2, 40.1],
    "AutoAttack": [0, 0.6, 1.0, 3.2, 6.1, 7.6, 9.8, 13.4, 15.6, 20.7],
    "FSA": [2, 10.2, 22.9, 35.7, 48.1, 58.3, 66.9, 73.6, 77.7, 82.8],
    "SIA": [69.1, 87.9, 95.5, 97.4, 99.0, 99.7, 100, 100, 100, 100]
}

vit = {
    "PGD": [1.9, 7.0, 16.2, 26.1, 37.9, 45.8, 53.5, 58.6, 62.1, 68.8],
    "CW": [0, 0.6, 4.5, 7.6, 9.9, 15.6, 19.4, 24.2, 29.9, 35.7],
    "AutoAttack": [0, 0.6, 0.9, 3.2, 6.1, 7.6, 9.6, 13.7, 16.9, 20.7],
    "FSA": [1.9, 6.7, 15.9, 25.4, 37.9, 47.8, 54.5, 60.2, 65.3, 69.4],
    "SIA": [29.0, 49.1, 61.5, 68.2, 73.9, 76.4, 79.9, 83.8, 86.6, 87.9]
}

# Plot styling
plt.figure(figsize=(10, 4))
markers = ['o', 's', '^', 'D', 'x']
linestyles = ['-', '--', '-.', ':', '-']
methods = list(resnet50.keys())

# Subplot 1: ResNet-50
plt.subplot(1, 2, 1)
for i, method in enumerate(methods):
    plt.plot(
        x,
        resnet50[method],
        marker=markers[i],
        linestyle=linestyles[i],
        label=method
    )
plt.title("ResNet-50", fontsize=11)
plt.xlabel("1 - SSIM", fontsize=10)
plt.ylabel("ASR (%)", fontsize=10)
plt.grid(True)

# Subplot 2: ViT-B/16
plt.subplot(1, 2, 2)
for i, method in enumerate(methods):
    plt.plot(
        x,
        vit[method],
        marker=markers[i],
        linestyle=linestyles[i],
        label=method
    )
plt.title("ViT-B/16", fontsize=11)
plt.xlabel("1 - SSIM", fontsize=10)
plt.grid(True)

# Unified legend
plt.legend(loc='upper left', fontsize=8)
plt.tight_layout()

# Export high-resolution figure
plt.savefig(r"asr_vs_ssim.pdf", dpi=300, bbox_inches='tight')
# plt.show()