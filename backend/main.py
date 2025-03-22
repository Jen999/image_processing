from fastapi import FastAPI, File, UploadFile, Form
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import io
import cv2
import numpy as np
import os
import requests

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# Initialize FastAPI app
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

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Florence-2 Model ONCE at startup
florence_model_name = "microsoft/Florence-2-large"
print("Loading Florence-2 model... (this may take time)")
florence_model = AutoModelForCausalLM.from_pretrained(
    florence_model_name, trust_remote_code=True
).to(device)
florence_processor = AutoProcessor.from_pretrained(
    florence_model_name, trust_remote_code=True, use_fast=True
)
print("Florence Model loaded successfully!")

# Load Qwen-2.5 Model
qwen_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
print("Loading Qwen-2.5 model... (this may take some time)")
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    qwen_model_name, torch_dtype="auto", device_map="auto"
)
qwen_processor = AutoProcessor.from_pretrained(qwen_model_name, use_fast=True)
print("Qwen Model loaded successfully!")


# Function to process image and generate OCR text using Florence-2 model
def generate_florence(
    image: Image.Image, task_prompt: str, model, processor, max_new_tokens=512
):
    print(f"Processing image with prompt using Florence-2: {task_prompt}")

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


# Function to process image with custom prompt and generate OCR text using Qwen-2.5 model
def generate_qwen(
    image: Image.Image, task_prompt: str, model, processor, max_new_tokens=128
):
    print(f"Processing image with prompt using Qwen-2.5: {task_prompt}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": task_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Trim input tokens from output
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode final text
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0] if output_text else ""


# Function to draw bounding boxes (for OCR with region)
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


# API Endpoint to Handle OCR Requests
@app.post("/ocr")
async def process_ocr(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model: str = Form("florence"),
):
    print(f"Received OCR request using model: {model} with prompt: {prompt}")

    # Read and process the image
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Check model to use for OCR
    if model == "florence":
        ocr_result = generate_florence(
            image, prompt, florence_model, florence_processor
        )

        if prompt == "<OCR>":
            extracted_text = ocr_result.get("<OCR>", "")
            return {"ocrText": extracted_text}

        # If the user requests region-based OCR, include bounding boxes
        elif prompt == "<OCR_WITH_REGION>":
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

        elif prompt == "<CAPTION>":
            extracted_text = ocr_result.get("<CAPTION>", "")
            return {"ocrText": extracted_text}

        elif prompt == "<DETAILED_CAPTION>":
            extracted_text = ocr_result.get("<DETAILED_CAPTION>", "")
            return {"ocrText": extracted_text}

        elif prompt == "<MORE_DETAILED_CAPTION>":
            extracted_text = ocr_result.get("<MORE_DETAILED_CAPTION>", "")
            return {"ocrText": extracted_text}

        else:
            return {"ocrText": ocr_result}

    else:
        ocr_result = generate_qwen(image, prompt, qwen_model, qwen_processor)
        return {"ocrText": ocr_result}
