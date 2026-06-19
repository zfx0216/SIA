import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

# ===========================
# Device
# ===========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===========================
# Image Preprocessing
# ===========================
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

inv_normalize = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
])

# ===========================
# Path Configuration
# ===========================
index_file_absolute_path = r"..."
weight_file_absolute_path = r"..."
actual_images_folder_absolute_path = r"..."
output_directory = r"..."

os.makedirs(output_directory, exist_ok=True)

# ===========================
# Hyperparameters
# ===========================
alpha = 0.001
T_total = 500
sparsity_ratio = 1.5
p_threshold = 100 - sparsity_ratio
num_images_test = 50

# ===========================
# Load Class Index & Model
# ===========================
with open(index_file_absolute_path, "r") as f:
    class_indict = json.load(f)

model = resnet50(num_classes=1000).to(device)
model.load_state_dict(torch.load(weight_file_absolute_path, map_location=device, weights_only=True))
model.eval()


# ===========================
# Core Function: Loss Fy(x) & Gradient Calculation
# ===========================
def get_loss_and_grad(img_tensor, true_label):
    img_tensor = img_tensor.clone().requires_grad_(True)
    output = model(img_tensor)
    loss = -output[0, true_label]

    grad = torch.autograd.grad(loss, img_tensor)[0]
    return loss.item(), grad


def get_sparse_mask(grad):
    grad_abs = torch.abs(grad).detach().cpu().numpy()
    threshold = np.percentile(grad_abs, p_threshold)
    mask = (grad_abs > threshold).astype(np.float32)
    return torch.from_numpy(mask).to(device)


# ===========================
# Single Image SIA Iteration & Metric Recording
# ===========================
def run_sia(img_tensor, true_label, T=500, lr=0.001):
    x_adv = img_tensor.clone()
    loss_list = []
    grad_norm_list = []
    cum_energy_list = []
    cum_energy = 0.0

    _, grad = get_loss_and_grad(x_adv, true_label)
    mask = get_sparse_mask(grad)

    for t in range(T):
        loss, grad = get_loss_and_grad(x_adv, true_label)
        loss_list.append(loss)

        grad_norm = torch.norm(grad, p=1).item()
        grad_norm_list.append(grad_norm)

        delta = lr * grad * mask
        x_adv = x_adv - delta

        delta_rgb = inv_normalize(delta)
        energy = torch.sum(delta_rgb ** 2).item()
        cum_energy += energy
        cum_energy_list.append(cum_energy)

    return loss_list, grad_norm_list, cum_energy_list


# ===========================
# Batch Execution
# ===========================
image_files = [f for f in os.listdir(actual_images_folder_absolute_path)
               if f.endswith(('png', 'jpg', 'jpeg'))]
image_files = image_files[:num_images_test]

all_loss = []
all_grad_norm = []
all_energy = []

print("Start SIA convergence experiment...")
start = time.time()

for idx, img_file in enumerate(image_files):
    img_path = os.path.join(actual_images_folder_absolute_path, img_file)
    img = Image.open(img_path).convert("RGB")
    img_tensor = data_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        true_label = model(img_tensor).argmax(1).item()

    loss, gn, energy = run_sia(img_tensor, true_label, T=T_total, lr=alpha)

    all_loss.append(loss)
    all_grad_norm.append(gn)
    all_energy.append(energy)

    print(f"[{idx + 1}/{num_images_test}] Completed: {img_file}")

end = time.time()
print(f"Total runtime: {end - start:.2f}s")

# ===========================
# Mean & Std Calculation
# ===========================
all_loss = np.array(all_loss)
all_grad_norm = np.array(all_grad_norm)
all_energy = np.array(all_energy)

loss_mean = all_loss.mean(axis=0)
loss_std = all_loss.std(axis=0)

grad_mean = all_grad_norm.mean(axis=0)
grad_std = all_grad_norm.std(axis=0)

energy_mean = all_energy.mean(axis=0)
energy_std = all_energy.std(axis=0)

iters = np.arange(1, T_total + 1)

# ===========================
# Visualization
# ===========================
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

# Objective value curve
plt.figure(figsize=(5, 4))
plt.plot(iters, loss_mean, 'b-', linewidth=2, label='Average')
plt.fill_between(iters, loss_mean - loss_std, loss_mean + loss_std, alpha=0.2, color='blue')
plt.xlabel("Iteration t")
plt.ylabel(r"$F_y(x_t)$")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, "obj_decrease.pdf"), bbox_inches='tight')
plt.close()

# Gradient L1 norm curve
plt.figure(figsize=(5, 4))
plt.plot(iters, grad_mean, 'r-', linewidth=2, label='Average')
plt.fill_between(iters, grad_mean - grad_std, grad_mean + grad_std, alpha=0.2, color='red')
plt.xlabel("Iteration t")
plt.ylabel(r"$\|\nabla F_y(x_t)\|_1$")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, "grad_norm.pdf"), bbox_inches='tight')
plt.close()

# Cumulative perturbation energy curve
plt.figure(figsize=(5, 4))
plt.plot(iters, energy_mean, 'g-', linewidth=2, label='Average')
plt.fill_between(iters, energy_mean - energy_std, energy_mean + energy_std, alpha=0.2, color='green')
plt.xlabel("Iteration t")
plt.ylabel("Cumulative perturbation energy")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, "energy.pdf"), bbox_inches='tight')
plt.close()

print("\nAll tasks finished! Figures saved to:", output_directory)
print(f"Final objective mean: {loss_mean[-1]:.4f} ± {loss_std[-1]:.4f}")
print(f"Final cumulative energy mean: {energy_mean[-1]:.4f} ± {energy_std[-1]:.4f}")