🧠 AI PCB Defect Detection & Classification System
📌 Project Overview

This project focuses on building an automated Printed Circuit Board (PCB) defect detection and classification system using image processing and deep learning techniques.

The system compares a defect-free (template) PCB image with a test image to detect defects and uses a trained Convolutional Neural Network (CNN) to classify them.

A web-based frontend allows users to upload images and view annotated results with highlighted defects.

🎯 Objectives
Detect and localize PCB defects using image subtraction techniques
Classify detected defects into predefined categories using CNN
Build and evaluate a deep learning model for accurate classification
Develop a frontend interface for image upload and visualization
Integrate backend processing pipeline for inference
Export annotated results and logs
🚀 Key Features
🔍 Defect Detection using image subtraction & thresholding
📦 ROI Extraction using contour detection
🧠 Deep Learning Classification (EfficientNet / CNN)
🌐 Web Interface for easy interaction
📊 Evaluation Metrics (Accuracy, Confusion Matrix)
💾 Export Results (Images + CSV logs)
🧩 Project Modules
1. Image Processing (OpenCV)
Image alignment and preprocessing
Image subtraction (template vs test)
Thresholding (Otsu’s method)
Noise filtering
2. Contour Detection & ROI Extraction
Detect defect regions
Extract bounding boxes
Crop defect areas for training
3. Deep Learning Model
Transfer learning (EfficientNet / CNN)
Image resizing (128x128)
Data augmentation
Training using Adam optimizer
4. Frontend
Built using Streamlit / HTML, CSS, JS
Upload template & test images
Display annotated output
5. Backend
Modular pipeline for:
Image preprocessing
ROI extraction
Model inference
Returns labeled images
6. Testing & Optimization
Model evaluation on unseen data
Performance tuning
7. Export & Reporting
Download annotated images
Export prediction logs (CSV)
8. Documentation & Presentation
README & technical docs
Demo video / slides
📊 Dataset
DeepPCB Dataset
Contains:
Template (defect-free) images
Test images with defects
🗓️ Project Timeline
🔹 Milestone 1 (Weeks 1–2): Image Processing
Dataset preparation
Image subtraction
Defect mask generation
🔹 Milestone 2 (Weeks 3–4): Model Training
Train CNN model
Evaluate accuracy & loss
Generate confusion matrix
🔹 Milestone 3 (Weeks 5–6): Integration
Build frontend UI
Integrate backend pipeline
🔹 Milestone 4 (Weeks 7–8): Finalization
Testing & optimization
Export features
Documentation
📈 Evaluation Criteria
Milestone	Focus Area	Metric	Goal
1	Image Processing	Mask quality	Detect all defects
2	Model	Accuracy	≥ 95%
3	UI	Response Time	≤ 3 sec
4	Final System	Export & Docs	Fully working
🛠️ Tech Stack
Area	Tools
Image Processing	OpenCV, NumPy
Deep Learning	PyTorch / TensorFlow
Dataset	DeepPCB
Frontend	Streamlit / HTML, CSS, JS
Backend	Python
Evaluation	Accuracy, Loss, Confusion Matrix
Export	CSV, Images, PDF
⚙️ Installation & Setup
# Clone the repository
git clone https://github.com/your-username/pcb-defect-detection.git

# Navigate to project folder
cd pcb-defect-detection

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
🖥️ Usage
Upload:
Template (defect-free) image
Test image
Click Detect Defects
View:
Highlighted defects
Classified labels
Download results (image + CSV)
📌 Expected Outcomes
Accurate defect detection using image processing
High classification accuracy (≥95%)
Fully functional web application
Exportable results for analysis
🔮 Future Enhancements
Real-time defect detection using camera input
Deployment on cloud (AWS / Azure)
Mobile-friendly interface
Support for multiple PCB datasets
Advanced models (Vision Transformers)
👩‍💻 Author
MCA Student | Software Engineer | AI Enthusiast
📜 License

This project is for academic and research purposes.
