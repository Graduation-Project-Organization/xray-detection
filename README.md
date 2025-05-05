# AI X-Ray Detection API

A lightweight Flask REST API that classifies chest X-ray images as **Normal** or **Pneumonia** using a deep learning model (`model.h5`).

#### 📦 Dependencies
`pip install flask tensorflow pillow flask-cors`

#### 🚀 Running the Application
`python app.py`

### 🌐 API Endpoints
#### 1. Check Server Status
Method (`GET`) : http://localhost:5000/

#### 2. Predict X-Ray
Method (`POST`) : http://localhost:5000/ai-xray-detection

- Notes
    - Upload an image file in the body (e.g., .jpg, .png)
    - Success Response: `{"result": "Pneumonia"}`


#### ❌ Errors
- Missing file: `{ "error": "No image uploaded" }`
- Invalid file type (e.g., .txt, corrupted image): `{ "error": "Invalid image file" }`
