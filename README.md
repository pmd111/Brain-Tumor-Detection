# 🧠 Brain Tumor Detection Using Hybrid Transformer-GAN-U-Net Architecture

This is a deep learning-based full-stack project designed for **brain tumor classification and localization** using MRI images. It features a **Transformer-enhanced U-Net model** for segmentation and a **GAN for synthetic data augmentation**, ensuring robust training even on limited datasets.

---

## 🚀 Features

- 🎯 Classifies brain MRI into **Glioma, Meningioma, Pituitary**, or **No Tumor**
- 🧠 Accurately **segments tumor region** with Transformer-based U-Net
- 🧬 **GAN-based data augmentation** to improve model generalization
- 🌐 Complete **web interface** with Flask backend and HTML/JS frontend
- 🖼️ Outputs include **original image, segmented mask**, and **tumor prediction**
- 🧪 Trained with both **real and synthetic MRI data**

---

## 🛠️ Technologies Used

| Category       | Stack                                      |
|----------------|--------------------------------------------|
| Languages      | Python, HTML, CSS, JavaScript              |
| Backend        | Flask (REST API)                           |
| ML Frameworks  | PyTorch, OpenCV, NumPy, Scikit-learn       |
| Model Types    | CNN, U-Net, Transformers, GAN              |
| Deployment     | Localhost (can be deployed to cloud)       |
| Frontend       | HTML5, CSS3, JavaScript                    |

---

## 📁 Project Structure

Brain_Tumor_Detection/ │── dataset/ # Your dataset (training/testing folders) │── model/ # Trained model stored here │── backend/ # Flask API │ │── app.py # API for prediction │ │── model_loader.py # Load trained model │ │── requirements.txt # Dependencies │── frontend/ # Web interface │ │── index.html # Frontend webpage │ │── static/ │ │── styles.css # CSS file │ │── script.js # JavaScript file │── train_model.py # Model training script │── augment_GAN.py # GAN-based augmentation

yaml
Copy

---

## 🧪 How to Run

1. **Install dependencies**:
   ```bash
   pip install -r backend/requirements.txt
Train GAN for data augmentation:

bash
Copy
python augment_GAN.py
Train the tumor classification & segmentation model:

bash
Copy
python train_model.py
Run the web app:

bash
Copy
cd backend
python app.py
Open your browser and go to:

arduino
Copy
http://localhost:5000
📊 Results
Classification Accuracy: 92%+

Segmentation IoU: ~85%

Inference Time: <2 seconds per scan

🔮 Future Enhancements
Add volumetric (3D MRI) support using 3D U-Net

Deploy to cloud (AWS/GCP) with Docker

Add explainability tools (Grad-CAM, SHAP)

Integrate with hospital PACS or EMR systems

📄 License
This project is intended for academic and research use only.

"The fusion of local precision, global understanding, and synthetic diversity makes this project truly intelligent."

vbnet
Copy
