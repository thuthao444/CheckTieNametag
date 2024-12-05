from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import os
from PIL import Image
import pandas as pd
import tempfile

OUTPUT_DIR1 = "./CheckTie/OutputTie"
OUTPUT_DIR2 = "./CheckNameTag/OutputCheckTag"
os.makedirs(OUTPUT_DIR1, exist_ok=True)
os.makedirs(OUTPUT_DIR2, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

model1 = YOLO("./CheckTie/yolov8l-seg.pt")
model2 = YOLO("./CheckNameTag/checkpoints/best.pt")

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

    # Lưu tạm hình ảnh
    temp_path = "temp_segmented_image.png"
    cv2.imwrite(temp_path, segmented_image)

    return temp_path

def calculate_histogram(image):
    # Đọc ảnh đã segment và chuyển đổi thành ảnh BGR
    image = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    # Chỉ lấy phần RGB, bỏ qua alpha
    alpha_channel = image[:, :, 3]
    mask = (alpha_channel == 255).astype(np.uint8)
    image_rgb = image[:, :, :3]  # Chỉ lấy phần RGB
    masked_img = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    #image_rgb = cv2.resize(masked_img, (256, 256))
    lab_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2Lab)
    hist = cv2.calcHist([lab_image], [1], mask, [256], [0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return hist.flatten()

def compare_histogram(hist1, hist2):
   similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
   return similarity

def calculate_mean_hist_sample():
    sample_folder = './CheckTie/TieSampleData'
    sample_images = os.listdir(sample_folder)
    sample_paths = [os.path.join(sample_folder, img) for img in sample_images if img.endswith(('.jpg', '.png', '.jpeg'))]

    sample_histograms = []
    for sample_path in sample_paths:
        tie = model1(source=sample_path, classes=[27])
        mask = tie[0].masks
        image_segment = tie_segment(sample_path, mask)
        sample_histograms.append(calculate_histogram(image_segment))

    mean_hist = np.mean(sample_histograms, axis=0)
    print("calculate mean hist")

    return mean_hist

@app.post("/tie")
async def check_tie(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="File is empty or not provided")

        file_bytes = np.frombuffer(contents, np.uint8)

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image_file:
            temp_image_path = temp_image_file.name
            cv2.imwrite(temp_image_path, image)

        mean_hist = calculate_mean_hist_sample()

        results = model1(source=image, classes=[27])

        detections = results[0].boxes.xyxy.cpu().numpy()  
        if len(detections) > 0:
            mask = results[0].masks
            segment_image = tie_segment(temp_image_path, mask)
            tie_hist = calculate_histogram(segment_image)
            similary = compare_histogram(tie_hist, mean_hist)

            os.remove(temp_image_path)
            os.remove(segment_image)

            if similary > 0.5:
                boxes = []
                for box in detections:
                    x1, y1, x2, y2 = map(int, box[:4])  
                    boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
                img = results[0].plot()
                output = Image.fromarray(img[..., ::-1])
                output_filename = f"output_{file.filename}.jpg"
                output_path = os.path.join(OUTPUT_DIR1, output_filename)
                output.save(output_path)
                return {"tie_boxes": boxes}
            else:
                return {"message": "not tie sample"}
        else:
            return {"message": "no tie"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/nametag")
async def check_nametag(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="File is empty or not provided")

        file_bytes = np.frombuffer(contents, np.uint8)

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        results = model2(source=image)

        detections = results[0].boxes.xyxy.cpu().numpy()  
        if len(detections) > 0:
            boxes = []
            for box in detections:
                x1, y1, x2, y2 = map(int, box[:4])  
                boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
            img = results[0].plot()
            output = Image.fromarray(img[..., ::-1])
            output_filename = f"output_{file.filename}.jpg"
            output_path = os.path.join(OUTPUT_DIR2, output_filename)
            output.save(output_path)
            return {"nametag_boxes": boxes}
        else:
            return {"message": "no nametag"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")