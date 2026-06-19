import os
import json
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_tiny
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

index_file_absolute_path = r"..."
weight_file_absolute_path = r"..."

actual_images_folder_absolute_path = r"..."
output_directory = r"..."

os.makedirs(output_directory, exist_ok=True)

iteration_step_size = 0.001

ratio = 97.5

max_num_iterative = 100
ssim_threshold = 0.9999


with open(index_file_absolute_path, "r") as f:
    class_indict = json.load(f)


model = convnext_tiny(num_classes=1000).to(device)
model.load_state_dict(torch.load(weight_file_absolute_path, map_location=device, weights_only=True))
model.eval()


@torch.no_grad()
def predict_tensor(img_tensor):
    output = model(img_tensor)
    prob = torch.softmax(output, dim=1)
    return torch.argmax(prob, dim=1).item(), output

def calculate_gradient_tensor(img_tensor, label):
    img_tensor.requires_grad_(True)
    output = model(img_tensor)
    target_score = output[0, label]
    grad = torch.autograd.grad(target_score, img_tensor)[0]
    grad = -grad
    return grad.squeeze(0).cpu().numpy()

def divide_matrix(input_matrix, level):
    threshold = np.percentile(input_matrix, level)
    return (input_matrix > threshold).astype(np.uint8)

def apply_pixel_change(img_np, grad_np, mask_np):
    delta = np.sign(grad_np) * mask_np
    img_np = img_np + delta.transpose(1, 2, 0)
    return np.clip(img_np, 0, 255)

def compute_ssim(original_np, adversarial_np):
    return ssim(original_np, adversarial_np, win_size=3)


image_files = [f for f in os.listdir(actual_images_folder_absolute_path) if f.endswith('.png')]

image_num = 0
success_num = 0
start_time = time.time()

for image_file in image_files:
    print(f"{image_file}")
    image_num += 1
    img_path = os.path.join(actual_images_folder_absolute_path, image_file)

    img_pil = Image.open(img_path).convert("RGB").resize((224, 224))
    img_original_np = np.array(img_pil).astype(np.float64)
    img_np = img_original_np.copy()

    img_original_uint8 = img_original_np.astype(np.uint8)
    img_tensor = data_transform(img_pil).unsqueeze(0).to(device)
    actual_label, _ = predict_tensor(img_tensor)

    grad = calculate_gradient_tensor(img_tensor, actual_label)
    fixed_mask = divide_matrix(np.abs(grad), ratio)

    flag = 0
    num_iter = 0

    while flag == 0 and num_iter <= max_num_iterative:
        num_iter += 1
        print(f"{num_iter}")

        img_tensor = data_transform(Image.fromarray(img_np.astype(np.uint8))).unsqueeze(0).to(device)
        grad = calculate_gradient_tensor(img_tensor, actual_label)

        img_np = apply_pixel_change(img_np, grad, fixed_mask)

        img_tensor = data_transform(Image.fromarray(img_np.astype(np.uint8))).unsqueeze(0).to(device)
        pred_label, _ = predict_tensor(img_tensor)

        current_adv_uint8 = img_np.astype(np.uint8)
        current_ssim = compute_ssim(img_original_uint8, current_adv_uint8)
        print(f"SSIM = {current_ssim:.5f}")

        if pred_label != actual_label:
            print("success")
            success_num += 1
            flag = 1
        if current_ssim < ssim_threshold:
            print("failed")
            flag = 1

        save_path = os.path.join(output_directory, image_file)
        Image.fromarray(current_adv_uint8).save(save_path)

end_time = time.time()