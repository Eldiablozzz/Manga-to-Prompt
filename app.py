import gradio as gr
from manga_ocr import MangaOcr
import easyocr
import numpy as np
from PIL import Image
import torch
import os
from transformers import AutoProcessor, AutoModelForCausalLM

# --- Global Models ---
MODELS = {
    "japanese": None,
    "english": None,
    "vision_model": None,
    "vision_processor": None
}

def load_models(lang, modules):
    global MODELS
    
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print(f"✅ Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        dtype = torch.float32
        print("⚠️ Running on CPU")

    # 1. Load OCR Models (Only if requested)
    if "OCR" in modules:
        if lang == "Japanese (manga-ocr)" and MODELS["japanese"] is None:
            print("Loading Japanese OCR...")
            MODELS["japanese"] = MangaOcr() 
        
        elif lang == "English (EasyOCR)" and MODELS["english"] is None:
            print("Loading English OCR...")
            MODELS["english"] = easyocr.Reader(['en'], gpu=(device == 'cuda'))

    # 2. Load Florence-2 (Only if requested)
    if "Visual Description" in modules and MODELS["vision_model"] is None:
        print("Loading Microsoft Florence-2...")
        model_id = "microsoft/Florence-2-large"
        
        MODELS["vision_processor"] = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        MODELS["vision_model"] = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            torch_dtype=dtype,
            attn_implementation="sdpa"
        ).to(device)
    
    return True

def generate_description(image, model, processor, device):
    """Generates a detailed visual description using Florence-2."""
    try:
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16 if device=='cuda' else torch.float32)
        
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
        return parsed_answer[prompt]
    except Exception as e:
        return f"Error: {e}"

# --- Smart Slicing Logic ---
def process_tall_image(image, lang_model, lang_type, vision_model, processor, modules):
    width, height = image.size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Settings
    SLICE_HEIGHT = 1500 
    
    full_output = []
    y = 0
    slice_count = 1
    
    while y < height:
        bottom = min(y + SLICE_HEIGHT, height)
        box = (0, y, width, bottom)
        slice_img = image.crop(box)
        
        scene_desc = ""
        text_content = ""

        # 1. Run Visual Inference (If Selected)
        if "Visual Description" in modules:
            # Convert to RGB for stability
            scene_desc = generate_description(slice_img.convert("RGB"), vision_model, processor, device)
        
        # 2. Run OCR (If Selected)
        if "OCR" in modules:
            try:
                if lang_type == "japanese":
                    text_content = lang_model(slice_img)
                else:
                    text_content = " ".join(lang_model.readtext(np.array(slice_img), detail=0, paragraph=True))
            except:
                pass 

        # 3. Construct Output
        # Only add segment if we actually found something
        has_content = (scene_desc and len(scene_desc) > 5) or (text_content and text_content.strip())
        
        if has_content:
            segment = f"--- [Segment {slice_count}] ---\n"
            if "Visual Description" in modules:
                segment += f"👀 VISUAL: {scene_desc}\n"
            if "OCR" in modules:
                segment += f"💬 TEXT: {text_content}\n"
            full_output.append(segment)

        y += SLICE_HEIGHT
        slice_count += 1
        
    return "\n".join(full_output)

# --- Main Processing Function ---
def process_manga_pages(files, language, modules, progress=gr.Progress()):
    if not files:
        return "No files uploaded.", None
    
    if not modules:
        return "⚠️ Error: You must select at least one module (OCR or Visual Description).", None

    load_models(language, modules)
    
    files.sort(key=lambda x: x.name)
    results = []
    
    progress(0, desc="Starting...")
    
    for i, file_path in enumerate(files):
        progress((i / len(files)), desc=f"Analyzing Page {i+1}...")
        
        try:
            Image.MAX_IMAGE_PIXELS = None 
            image = Image.open(file_path)
            
            extracted_data = ""
            
            # Identify which OCR model to pass (if any)
            ocr_model = None
            lang_key = "japanese" if "Japanese" in language else "english"
            
            if "OCR" in modules:
                ocr_model = MODELS[lang_key]

            extracted_data = process_tall_image(
                image, 
                ocr_model, 
                lang_key, 
                MODELS["vision_model"], 
                MODELS["vision_processor"],
                modules
            )
            
            page_result = f"=== PAGE {i+1}: {os.path.basename(file_path)} ===\n{extracted_data}\n\n"
            results.append(page_result)
            
        except Exception as e:
            results.append(f"[Error Page {i+1}]: {str(e)}\n")
            
    full_output = "\n".join(results)
    
    output_filename = "analysis_result.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(full_output)
        
    return full_output, output_filename

# --- Web Interface ---
with gr.Blocks(title="Custom AI Manga Analyst", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("# 👁️ Custom AI Manga Analyst")
    gr.Markdown("Select the features you need. Uncheck modules to save memory and speed up processing.")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                file_count="multiple", 
                file_types=["image"],
                label="Upload Webtoon/Manga Pages"
            )
            
            # --- NEW: Checkboxes for control ---
            modules_input = gr.CheckboxGroup(
                choices=["OCR", "Visual Description"],
                value=["OCR", "Visual Description"], # Default to both ON
                label="Active AI Modules"
            )
            
            lang_dropdown = gr.Dropdown(
                choices=["Japanese (manga-ocr)", "English (EasyOCR)"],
                value="English (EasyOCR)", 
                label="Select Language (For OCR)"
            )
            
            submit_btn = gr.Button("Analyze Manga", variant="primary")
        
        with gr.Column():
            text_output = gr.Textbox(label="Script Data", lines=25)
            download_btn = gr.File(label="Download .txt")

    submit_btn.click(
        fn=process_manga_pages, 
        inputs=[file_input, lang_dropdown, modules_input], 
        outputs=[text_output, download_btn]
    )

if __name__ == "__main__":
    app.launch(inbrowser=True)