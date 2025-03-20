from fastapi import FastAPI, File, UploadFile, Form
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import io
import cv2
import numpy as np
import os
import requests

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# Step 1: Initialize FastAPI app
app = FastAPI()

# Serve the `backend/static/` folder at `/static`
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow frontend to access backend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Step 2: Load Florence-2 Model ONCE at startup
model_name = "microsoft/Florence-2-large"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Florence-2 model... (this may take time)")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(
    device
)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
print("Model loaded successfully!")

# DeepSeek API Configurations
DEEPSEEK_API_URL = "https://api.deepseek.com"
DEEPSEEK_API_KEY = "sk-4e6b88d828b9485a945929f9873b5584"


# Step 3: Function to process image and generate OCR text
def generate_ocr(image: Image.Image, task_prompt: str, max_new_tokens=512):
    print(f"Processing image with prompt: {task_prompt}")

    # Convert image and prompt into tensors
    inputs = processor(images=image, text=task_prompt, return_tensors="pt").to(device)

    # Generate text from the model
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        num_beams=3,
        do_sample=False,
    )

    # Decode the generated text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text, task=task_prompt, image_size=image.size
    )

    # DeepSeek API
    # # Convert image to bytes
    # img_byte_arr = io.BytesIO()
    # image.save(img_byte_arr, format="PNG")
    # img_byte_arr = img_byte_arr.getvalue()

    # # Prepare the request payload
    # headers = {
    #     "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    #     "Content-Type": "application/octet-stream",
    # }
    # params = {"task_prompt": task_prompt}

    # # Make the request to DeepSeek API
    # response = requests.post(
    #     DEEPSEEK_API_URL, headers=headers, params=params, data=img_byte_arr
    # )

    # if response.status_code == 200:
    #     return response.json()
    # else:
    #     raise Exception(
    #         f"DeepSeek API request failed with status code {response.status_code}: {response.text}"
    #     )


# Step 4: Function to draw bounding boxes
def draw_bounding_boxes(image_bytes, ocr_result):
    print("Drawing bounding boxes on detected text regions...")

    # Convert image bytes to OpenCV format
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Get image dimensions
    height, width, _ = image.shape
    print(f"Image dimensions: {width}x{height}")

    # Extract bounding boxes and labels
    quad_boxes = ocr_result.get("quad_boxes", [])
    labels = ocr_result.get("labels", [])

    # Draw bounding boxes
    for box, label in zip(quad_boxes, labels):
        # Ensure coordinates are within valid range
        x_min = max(0, min(width, int(min(box[::2]))))  # Min X (Left)
        y_min = max(0, min(height, int(min(box[1::2]))))  # Min Y (Top)
        x_max = max(0, min(width, int(max(box[::2]))))  # Max X (Right)
        y_max = max(0, min(height, int(max(box[1::2]))))  # Max Y (Bottom)

        # Draw rectangle around detected text
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Put label text above the bounding box
        cv2.putText(
            image,
            label,
            (x_min, max(20, y_min - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Save the output image
    backend_dir = os.path.dirname(os.path.abspath(__file__))  # Get backend folder path
    static_dir = os.path.join(backend_dir, "static")  # Path to static folder
    os.makedirs(static_dir, exist_ok=True)  # Create folder if missing

    output_filename = "annotated_image.jpg"
    output_path = os.path.join(static_dir, output_filename)
    cv2.imwrite(output_path, image)

    print(f"Annotated image saved as {output_path}")
    return f"static/{output_filename}"


# Step 5: API Endpoint to Handle OCR Requests
@app.post("/ocr")
async def process_ocr(image: UploadFile = File(...), prompt: str = Form(...)):
    print("Received OCR request...")

    # Read and process the image
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Perform OCR
    ocr_result = generate_ocr(image, prompt)

    # If the user requests region-based OCR, include bounding boxes
    if prompt == "<OCR>":
        extracted_text = ocr_result.get("<OCR>", "")
        return {"ocrText": extracted_text}

    if prompt == "<OCR_WITH_REGION>":
        extracted_text = ocr_result.get("<OCR_WITH_REGION>", "")
        bbox_image_path = draw_bounding_boxes(
            image_bytes, ocr_result["<OCR_WITH_REGION>"]
        )
        labels = ocr_result["<OCR_WITH_REGION>"].get("labels", [])

        return {
            "ocrText": extracted_text,
            "annotatedImage": bbox_image_path,
            "labels": labels,
        }

    if prompt == "<CAPTION>":
        extracted_text = ocr_result.get("<CAPTION>", "")
        return {"ocrText": extracted_text}

    if prompt == "<DETAILED_CAPTION>":
        extracted_text = ocr_result.get("<DETAILED_CAPTION>", "")
        return {"ocrText": extracted_text}

    if prompt == "<MORE_DETAILED_CAPTION>":
        extracted_text = ocr_result.get("<MORE_DETAILED_CAPTION>", "")
        return {"ocrText": extracted_text}

    return
