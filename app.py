import streamlit as st
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
import easyocr
from paddleocr import PaddleOCR

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced OCR with Interactive Selection",
    page_icon="🎯",
    layout="wide"
)

# --- Model Loading ---
@st.cache_resource
def load_grounding_model():
    processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("google/owlvit-base-patch32")
    return processor, model

@st.cache_resource
def load_trocr_model():
    model_name = "microsoft/trocr-base-printed"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return processor, model

@st.cache_resource
def load_easyocr_reader():
    return easyocr.Reader(['en'], gpu=False)

@st.cache_resource
def load_paddleocr_reader():
    return PaddleOCR(use_angle_cls=True, lang='en')

# --- Helper Functions ---
def draw_numbered_boxes(image, boxes):
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except IOError:
        font = ImageFont.load_default(size=32)

    for i, box in enumerate(boxes):
        box_coords = box.tolist()
        draw.rectangle(box_coords, outline="lime", width=5)
        draw.rectangle([box_coords[0], box_coords[1], box_coords[0] + 40, box_coords[1] + 40], fill="lime")
        draw.text((box_coords[0] + 5, box_coords[1] + 5), str(i + 1), fill="black", font=font)
    return image_copy

def run_trocr_ocr(image, processor, model):
    # Preprocess: resize, enhance contrast (keep RGB)
    target_size = (384, 384)
    img = image.resize(target_size, Image.BICUBIC)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    inputs = processor(images=img, text="", return_tensors="pt")
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def run_easyocr_ocr(image, reader):
    # EasyOCR expects numpy array
    img_np = np.array(image)
    result = reader.readtext(img_np, detail=0)
    return " ".join(result)

def run_paddleocr_ocr(image, reader):
    # PaddleOCR expects numpy array (BGR)
    img_np = np.array(image.convert("RGB"))
    result = reader.ocr(img_np, cls=True)
    text = " ".join([line[1][0] for line in result[0]])
    return text

# --- Main App Interface ---
st.title("🎯 Advanced OCR with Interactive Selection")
st.write(
    "**Step 1:** The app uses OWL-ViT to find all objects matching your prompt. "
    "**Step 2:** See numbered boxes on the image, then select a number to run advanced OCR (Florence-2) on just that object."
)


with st.spinner("Loading AI Models (This will take a moment)..."):
    grounding_processor, grounding_model = load_grounding_model()
    trocr_processor, trocr_model = load_trocr_model()
    easyocr_reader = load_easyocr_reader()
    paddleocr_reader = load_paddleocr_reader()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    prompt_col, threshold_col = st.columns(2)
    with prompt_col:
        text_prompt = st.text_input("Object to find:", "a billboard")
    with threshold_col:
        confidence_threshold = st.slider("Object Detection Confidence", 0.05, 1.0, 0.1, 0.05)

    original_image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Step 1: Finding and numbering objects..."):
        inputs = grounding_processor(text=[text_prompt], images=original_image, return_tensors="pt")
        outputs = grounding_model(**inputs)
        target_sizes = torch.tensor([original_image.size[::-1]])
        results = grounding_processor.post_process_object_detection(
            outputs=outputs,
            threshold=confidence_threshold,
            target_sizes=target_sizes
        )[0]
        boxes = results["boxes"]

    st.subheader("Step 1: Found Objects")
    if len(boxes) > 0:
        numbered_image = draw_numbered_boxes(original_image, boxes)
        st.image(numbered_image, caption=f"Found {len(boxes)} object(s). Select one below to analyze.")
    else:
        st.warning("No objects found. Try a different prompt or lower the confidence threshold.")
        st.image(original_image, caption="Original Image")

    if len(boxes) > 0:
        st.divider()
        st.subheader("Step 2: Select an Object to Read")
        selected_box_number = st.number_input(
            f"Select an object number (1 to {len(boxes)}):",
            min_value=1,
            max_value=len(boxes),
            step=1
        )
        selected_box_index = selected_box_number - 1
        selected_box_coords = boxes[selected_box_index].tolist()
        # Add margin to crop (10% of width/height)
        x0, y0, x1, y1 = selected_box_coords
        w, h = original_image.size
        margin_x = int(0.1 * (x1 - x0))
        margin_y = int(0.1 * (y1 - y0))
        x0 = max(0, x0 - margin_x)
        y0 = max(0, y0 - margin_y)
        x1 = min(w, x1 + margin_x)
        y1 = min(h, y1 + margin_y)
        cropped_image = original_image.crop([x0, y0, x1, y1])

        st.image(cropped_image, caption=f"Selected Object #{selected_box_number}")
        st.write("#### Extracted Text by Model:")
        model_options = ["TrOCR", "EasyOCR", "PaddleOCR"]
        selected_models = st.multiselect("Select OCR models to run:", model_options, default=model_options)
        results = {}
        if "TrOCR" in selected_models:
            with st.spinner("Running TrOCR..."):
                results["TrOCR"] = run_trocr_ocr(cropped_image, trocr_processor, trocr_model)
        if "EasyOCR" in selected_models:
            with st.spinner("Running EasyOCR..."):
                results["EasyOCR"] = run_easyocr_ocr(cropped_image, easyocr_reader)
        if "PaddleOCR" in selected_models:
            with st.spinner("Running PaddleOCR..."):
                results["PaddleOCR"] = run_paddleocr_ocr(cropped_image, paddleocr_reader)
        for model_name, text in results.items():
            st.markdown(f"**{model_name}:**")
            st.code(text, language=None)