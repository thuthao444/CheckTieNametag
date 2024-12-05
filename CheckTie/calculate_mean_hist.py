import numpy as np
import cv2
import os
from ultralytics import YOLO

model1 = YOLO("./yolov8l-seg.pt")

def calculate_histogram(image):
    image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    alpha_channel = image[:, :, 3]
    mask = (alpha_channel == 255).astype(np.uint8)
    image_rgb = image[:, :, :3]  
    lab_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2Lab)
    hist = cv2.calcHist([lab_image], [1], mask, [256], [0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()

def tie_segment(original_path, mask):
    original_image = cv2.imread(original_path)
    mask_tensor = mask.data
    binary_mask = mask_tensor[0].numpy().astype(np.uint8)
    binary_mask[binary_mask > 0] = 1
    binary_mask_resized = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    segmented_image = np.zeros((original_image.shape[0], original_image.shape[1], 4), dtype=np.uint8)
    segmented_image[..., :3] = original_image
    segmented_image[binary_mask_resized == 1, 3] = 255
    segmented_image[binary_mask_resized == 0, 3] = 0
    temp_path = "temp_segmented_image.png"
    cv2.imwrite(temp_path, segmented_image)
    return temp_path

def save_mean_hist(mean_hist, file_path):
    np.save(file_path, mean_hist)
    print(f"mean_hist đã được lưu vào {file_path}")

def calculate_mean_hist_sample():
    sample_folder = './TieSampleData'
    sample_images = os.listdir(sample_folder)
    sample_paths = [os.path.join(sample_folder, img) for img in sample_images if img.endswith(('.jpg', '.png', '.jpeg'))]

    sample_histograms = []
    for sample_path in sample_paths:
        tie = model1(source=sample_path, classes=[27])
        mask = tie[0].masks
        image_segment = tie_segment(sample_path, mask)
        sample_histograms.append(calculate_histogram(image_segment))

    mean_hist = np.mean(sample_histograms, axis=0)
    save_mean_hist(mean_hist, 'mean_hist.npy')

def compare_histogram(hist1, hist2):
   similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
   return similarity

if __name__ == "__main__":
    calculate_mean_hist_sample()

