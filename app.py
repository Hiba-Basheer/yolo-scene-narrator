import os
from collections import Counter

import cv2
import gradio as gr
from groq import Groq
import numpy as np
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# Config 
MODEL_PATH   = "yolov8n.pt"      # auto-downloaded on first run (~6 MB)
CONF_THRESH  = 0.40              # minimum detection confidence

# Load YOLO once at startup 
print("Loading YOLOv8 model …")
yolo = YOLO(MODEL_PATH)
print("YOLOv8 ready.")


# run YOLO on a PIL image
def run_yolo(pil_img: Image.Image, conf: float):
    """
    Returns:
        annotated_img  : PIL image with bounding boxes drawn
        detections     : list of dicts  {label, confidence, bbox}
        object_counts  : Counter  e.g. Counter({'person': 3, 'car': 1})
    """
    img_np = np.array(pil_img.convert("RGB"))
    results = yolo(img_np, conf=conf)[0]

    detections = []
    for box in results.boxes:
        label      = yolo.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "label":      label,
            "confidence": round(confidence, 2),
            "bbox":       [x1, y1, x2, y2],
        })

    # Draw boxes on a copy
    annotated = img_np.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = (46, 204, 113)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text  = f"{d['label']} {d['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    object_counts = Counter(d["label"] for d in detections)
    return Image.fromarray(annotated), detections, object_counts


# build a readable object summary for Gemini 
def build_object_summary(object_counts: Counter, detections: list) -> str:
    if not object_counts:
        return "No objects were detected in the image."

    lines = []
    for label, count in object_counts.most_common():
        confs = [d["confidence"] for d in detections if d["label"] == label]
        avg_conf = sum(confs) / len(confs)
        lines.append(f"  • {count}× {label}  (avg confidence {avg_conf:.0%})")

    return "Detected objects:\n" + "\n".join(lines)


# call Gemini
def call_gemini(api_key: str, object_summary: str, mode: str) -> str:
    client = Groq(api_key=api_key)

    mode_instructions = {
        "Scene narration": "You are a vivid scene narrator. Given the list of detected objects, write a natural, engaging 3–4 sentence description of what is likely happening in the scene. Be specific and imaginative.",
        "Security analysis": "You are a security analyst reviewing CCTV footage. Given the detected objects, write a concise security assessment: note any persons, vehicles, or unusual items. Flag anything that might warrant attention.",
        "Children's story": "You are a children's book author. Given the detected objects, write a fun, playful 3–4 sentence story suitable for young children. Use simple words and make it whimsical and delightful.",
        "News reporter": "You are a breaking-news TV reporter. Given the detected objects, deliver a dramatic 3–4 sentence live news report about the scene. Use urgent, broadcast-style language.",
    }

    system = mode_instructions.get(mode, mode_instructions["Scene narration"])

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": object_summary},
        ],
    )
    return response.choices[0].message.content.strip()


#  Detection summary markdown
def format_detection_table(detections: list) -> str:
    if not detections:
        return "**No objects detected.** Try lowering the confidence threshold."

    rows = ["| Object | Confidence | Bounding Box |",
            "|--------|-----------|--------------|"]
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        rows.append(f"| {d['label']} | {d['confidence']:.0%} | ({x1},{y1}) → ({x2},{y2}) |")
    return "\n".join(rows)


#  Main pipeline
def process(image, conf_thresh: float, narration_mode: str):
    #  Validate inputs
    if image is None:
        return None, "⚠️ Please upload an image.", "", ""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, "❌ GROQ_API_KEY not found in .env file.", "", ""

    pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image

    #  Step 1: YOLO detection
    try:
        annotated_img, detections, object_counts = run_yolo(pil_img, conf_thresh)
    except Exception as e:
        return None, f"❌ YOLO error: {e}", "", ""

    object_summary = build_object_summary(object_counts, detections)
    detection_md   = format_detection_table(detections)

    #  Step 2: Gemini narration
    if not detections:
        narration = "🔍 No objects detected — nothing to narrate. Try a different image or lower the confidence threshold."
    else:
        try:
            narration = call_gemini(api_key.strip(), object_summary, narration_mode)
        except Exception as e:
            narration = f"❌ Gemini error: {e}"

    #  Step 3: Stats summary
    total    = len(detections)
    unique   = len(object_counts)
    top_item = object_counts.most_common(1)[0][0] if object_counts else "—"
    stats_md = (
        f"**{total}** objects detected &nbsp;·&nbsp; "
        f"**{unique}** unique classes &nbsp;·&nbsp; "
        f"Most common: **{top_item}**"
    )

    return annotated_img, narration, detection_md, stats_md


#  Gradio UI
css = """
#title    { text-align: center; font-size: 2rem; font-weight: 700; margin-bottom: 0; }
#subtitle { text-align: center; color: #666; margin-bottom: 1.5rem; }
.narration-box textarea { font-size: 1.05rem !important; line-height: 1.7 !important; }
"""

with gr.Blocks(title="YOLOv8 Scene Narrator") as demo:

    gr.Markdown("# 🎯 YOLOv8 + Groq Scene Narrator", elem_id="title")
    gr.Markdown(
        "Upload any image → **YOLOv8** detects objects → **Groq** narrates the scene.",
        elem_id="subtitle"
    )

    with gr.Row():
        #  Left column: inputs
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Image",
                type="numpy",
                height=300,
            )
            with gr.Row():
                conf_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.40, step=0.05,
                    label="Detection Confidence Threshold",
                )
            narration_mode = gr.Dropdown(
                choices=["Scene narration", "Security analysis",
                         "Children's story", "News reporter"],
                value="Scene narration",
                label="Narration Style",
            )
            run_btn = gr.Button("🔍 Detect & Narrate", variant="primary", size="lg")

        #  Right column: outputs
        with gr.Column(scale=1):
            annotated_out = gr.Image(label="Detected Objects", height=300)
            stats_out     = gr.Markdown()
            narration_out = gr.Textbox(
                label="AI Narration",
                lines=5,
                interactive=False,
                elem_classes=["narration-box"],
            )
            detection_out = gr.Markdown(label="Detection Details")

    #  Examples
    gr.Markdown("### 💡 Try these sample prompts after uploading an image")
    gr.Markdown(
        "- Upload a street photo → try **Security analysis**\n"
        "- Upload a pet photo → try **Children's story**\n"
        "- Upload a sports photo → try **News reporter**"
    )

    #  Wire up
    run_btn.click(
        fn=process,
        inputs=[image_input, conf_slider, narration_mode],
        outputs=[annotated_out, narration_out, detection_out, stats_out],
    )

    #  Auto-run when image is uploaded
    image_input.change(
        fn=process,
        inputs=[image_input, conf_slider, narration_mode],
        outputs=[annotated_out, narration_out, detection_out, stats_out],
    )


if __name__ == "__main__":
    demo.launch(css=css, share=False)
