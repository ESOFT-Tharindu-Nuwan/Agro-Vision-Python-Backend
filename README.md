# AgroVision Cassava Disease Detection Backend API

<p align="center">
  <img src="" alt="AgroVision Logo" width="150"/>
</p>

A robust and scalable backend API built with **FastAPI** and **TensorFlow** to classify cassava leaf diseases from images. This API serves a pre-trained deep learning model to provide instant, AI-powered diagnoses — a vital tool for farmers in Sri Lanka to protect their crops and livelihoods.

---

## ✨ Features

- 🌿 **Cassava Disease Prediction** using a deep learning model.
- 📷 Accepts image input and returns predicted disease class + confidence.
- 🧠 Flags uncertain predictions as **unidentifiable**.
- ⚡ Built with **FastAPI** + **Uvicorn** for high-speed API performance.
- ☁️ Ready to deploy to AWS, Heroku, GCP, or your own server.

---

## 🚀 Quick Setup Guide

Use this section to clone the project and run it from scratch.

### ✅ Prerequisites

Make sure the following are installed on your machine:

- [Python 3.8+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)

---

### 🔗 Step 1: Clone the Project

```bash
git clone https://github.com/ESOFT-Tharindu-Nuwan/Agro-Vision-Python-Backend.git
cd agro-vision-back-end-python
```

---

### 🧪 Step 2: Create a Virtual Environment

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows CMD:**

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 📦 Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 📁 Step 4: Add the Trained Model

1. Create the `models` directory:

    ```bash
    mkdir models
    ```

2. Place your trained `.h5` model inside the `models/` directory.

   Example filename used in code:  
   `models/sri_lankan_cassava_model_20250627_021853.h5`

3. (Optional) If your file name is different, edit `main.py` to match it.

---

## ▶️ Running the API

To launch the API server in development mode:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You can now access:

- 🌐 API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- 🔎 Health Check: [http://localhost:8000/](http://localhost:8000/)

---

## 📂 Project Structure

```
agro-vision-back-end-python/
├── venv/                       # Virtual environment (ignored)
├── models/                     # Your trained model goes here
│   └── sri_lankan_cassava_model_20250627_021853.h5
├── main.py                     # FastAPI application
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## 🧪 API Reference

### `GET /`

Simple health check.

**Response:**

```json
{
  "message": "Cassava Disease Detection API is running!"
}
```

---

### `POST /predict`

Upload an image for prediction.

**Form-Data Field:**
- `file`: Upload an image (`.jpg`, `.jpeg`, `.png`)

**Success Response:**

```json
{
  "status": "success",
  "message": "Disease identified successfully.",
  "predicted_class": "CMD",
  "confidence": 0.9876,
  "all_probabilities": {
    "CMD": 0.9876,
    "CBSD": 0.005,
    "CBB": 0.003,
    "Healthy": 0.0044
  }
}
```

**Low Confidence Response:**

```json
{
  "status": "unidentifiable",
  "message": "Cannot confidently identify the leaf...",
  "predicted_class": "Healthy",
  "confidence": 0.65,
  "all_probabilities": {
    "CMD": 0.20,
    "CBSD": 0.15,
    "CBB": 0.05,
    "Healthy": 0.65
  }
}
```

**Invalid File Response:**

```json
{
  "detail": "Invalid file type. Please upload an image file (e.g., JPG, PNG)."
}
```

---

## ⚙️ Configuration

You can modify the `CONFIDENCE_THRESHOLD` inside `main.py` to fine-tune model sensitivity.

---

## 📄 License

This project is **open-source** and free to use.

---

## 🙋 Need Help?

For questions or contributions, feel free to open an [issue](https://github.com/ESOFT-Tharindu-Nuwan/Agro-Vision-Python-Backend.git/issues) or contact [Tharindu Nuwan](https://github.com/Tharindu-Nuwan).

---

