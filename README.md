# 🎯 YOLOv8 + Groq Scene Narrator

A computer vision + LLM project that detects objects in images using **YOLOv8**
and narrates the scene using **Groq + LLaMA 3.1**.

---

## 🧠 How It Works

```
Image → YOLOv8 (CV model) → Detected objects list → Groq LLM → Natural language narration
```

**The CV layer (YOLOv8):**
- Runs real object detection — not just an API call
- Returns bounding boxes, class labels, and confidence scores
- Uses the COCO dataset classes (80 objects: person, car, dog, chair, etc.)

**The LLM layer (Groq + LLaMA 3.1):**
- Receives the structured detection output
- Generates scene descriptions in 4 different styles
- Bridges computer vision output → human-readable language

---

## 🚀 Setup & Run

### 1. Clone this repository
```bash
git clone https://github.com/hibamb/yolo-scene-narrator
cd yolo-scene-narrator
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get your free Groq API key
Go to: https://console.groq.com
- Sign up with Google
- Click "API Keys" → Create key
- Create a `.env` file in the project folder and add:
```
GROQ_API_KEY=your_key_here
```

### 4. Run the app
```bash
python app.py
```

Open your browser at: **http://localhost:7860**

---

## 🎛️ Features

| Feature | Details |
|---------|---------|
| Object detection | YOLOv8n (nano) — fast, runs on CPU |
| Classes detected | 80 COCO classes (person, car, dog, laptop, etc.) |
| Confidence slider | Adjust detection sensitivity |
| Narration styles | Scene narration, Security analysis, Children's story, News reporter |
| Bounding boxes | Drawn directly on image with labels + confidence % |
| Detection table | Full list of every detected object |

---

## 📁 Project Structure

```
yolo-scene-narrator/
│
├── app.py               ← Main application (all logic + UI)
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
├── .env                 ← Your API key (never commit this)
├── .gitignore           ← Keeps .env out of GitHub
└── yolov8n.pt           ← Auto-downloaded on first run
```

---

## 🔧 Customisation Ideas

- **Swap the model**: Change `yolov8n.pt` to `yolov8s.pt` or `yolov8m.pt` for better accuracy
- **Add video support**: Use `cv2.VideoCapture()` for webcam/video input
- **Custom classes**: Train YOLOv8 on your own dataset (e.g. skin lesions, plant diseases)
- **Add more narration modes**: Medical report, wildlife documentary, sports commentary
- **Save results**: Export annotated image + narration as a PDF report

---

## 📊 What This Demonstrates

- ✅ Real computer vision pipeline (not just an API wrapper)
- ✅ Object detection with bounding box drawing (OpenCV)
- ✅ LLM prompt engineering with structured inputs
- ✅ Multi-model pipeline design (CV model → LLM)
- ✅ Gradio UI for live demo
- ✅ Production-ready code structure

---

## 🆓 Costs

- **YOLOv8**: 100% free and open source (Ultralytics)
- **Groq + LLaMA 3.1**: Free tier — 14,400 requests/day, no regional restrictions
- **Gradio**: Free and open source

Total cost: **₹0**# yolo-scene-narrator
