import os
import torch
import numpy as np
from transformers import pipeline
from diffusers.utils import load_image, make_image_grid
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Add this to handle the OpenMP error


def visualize_depth_map(depth_map, output_path):
    # 确保深度图为单通道2D数组
    if depth_map.dim() == 3 and depth_map.shape[0] == 3:  # 假设错误的形状是 (3, H, W)
        depth_map = depth_map[0]  # 取第一个通道，或根据实际情况选择适当的通道
    # 将PyTorch张量转换为NumPy数组，并确保其在CPU上
    depth_map = depth_map.squeeze().cpu().numpy()
    plt.style.use('default')  # Reset to default to avoid using the problematic style setting
    plt.figure(figsize=(10, 6))
    plt.imshow(depth_map, cmap='hot')  # 使用热图颜色映射
    plt.colorbar()  # 显示颜色条
    plt.title("Depth Map Visualization")
    plt.savefig(output_path)  # Save the figure
    plt.show()


def get_depth_map(image, depth_estimator):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    return depth_map


def load_image_from_file(image_path):
    """ Load an image from a file path. """
    with Image.open(image_path) as img:
        return img.convert('RGB')


def save_depth_tensor_to_png(depth_map, output_depth_path):
    # Ensure depth_map is on CPU and convert to numpy
    depth_map_np = depth_map.cpu().numpy()
    # Handle multiple channels by averaging (alternative methods might be required based on actual use-case)
    if depth_map_np.shape[0] == 3:
        # Average across the channels if it has 3 channels
        depth_map_np = np.mean(depth_map_np, axis=0)
    depth_map_np = (255 * (depth_map_np - np.min(depth_map_np)) / (np.max(depth_map_np) - np.min(depth_map_np))).astype(
        np.uint8)
    img = Image.fromarray(depth_map_np)

    img.save(output_depth_path, "PNG")


def process_images_from_folder(input_folder, output_folder, saved_depth_folder, depth_estimator, num_images):
    """ Process a fixed number of images from a folder, generate depth maps, and save them. """
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),  # Resize images if necessary
        transforms.ToTensor()
    ])

    images_processed = 0
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Assuming JPEG or PNG images
            image_path = os.path.join(input_folder, filename)
            image = load_image_from_file(image_path)
            # image = transform(image).unsqueeze(0)  # Add batch dimension
            depth_map = get_depth_map(image, depth_estimator)
            # output_path = os.path.join(output_folder, f"depth_{filename}")
            output_depth_path = os.path.join(saved_depth_folder, f"depth_{filename}")
            save_depth_tensor_to_png(depth_map, output_depth_path)
            images_processed += 1

            if images_processed >= num_images:
                break  # Stop after processing the fixed number of images


os.environ["http_proxy"] = "http://127.0.0.1:10809"
os.environ["https_proxy"] = "http://127.0.0.1:10809"

input_folder = r"/mnt/share/toky/Datasets/EndoDepth-Diffusion/EndoSlam-Unity/eval/"
output_folder = "saved_depth_visualization"
saved_data_root = r"saved_depth/"  # saved_depth/
output_depth_path = os.path.join(saved_data_root, "EndoSlam-Unity")  # 模型和测试数据集名字
os.makedirs(os.path.dirname(output_depth_path), exist_ok=True)
depth_estimator = pipeline("depth-estimation", model=r"/mnt/share/toky/LLMs/dpt-large/")
num_images = 300  # 要处理的图片张数
process_images_from_folder(input_folder, output_folder, output_depth_path, depth_estimator, num_images)
