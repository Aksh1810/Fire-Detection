# 🔥 Fire Detection System

An intelligent real-time fire and smoke detection system powered by YOLOv8 deep learning model. This project uses computer vision and artificial intelligence to detect fire hazards through webcam feeds or static images, providing immediate audio alerts when danger is detected.

## 🎯 Overview

This system leverages state-of-the-art YOLOv8 object detection to identify fire and smoke in real-time video streams. Designed for safety monitoring, the application can process live webcam feeds or analyze images, making it suitable for home security, industrial monitoring, or research purposes.

## ✨ Features

- **Real-Time Detection**: Monitors webcam feeds continuously for fire and smoke
- **Image Analysis**: Process static images for fire presence
- **Instant Alerts**: Audio beep notification when fire/smoke is detected
- **Custom Training**: Train your own models on custom fire datasets
- **GPU Acceleration**: Supports training on Google Colab with free GPUs
- **Lightweight**: Runs efficiently on CPU for real-time inference (~30 FPS)
- **Cross-Platform**: Works on macOS, Linux, and Windows

## 🛠️ Technology Stack

- **AI Model**: YOLOv8 (Ultralytics)
- **Framework**: PyTorch
- **Computer Vision**: OpenCV
- **Language**: Python 3.8+
- **Dataset Management**: Roboflow

## 📊 Performance

- **Inference Speed**: ~30-40ms per frame on CPU
- **Detection Classes**: Fire, Smoke
- **Model Size**: ~6MB (YOLOv8n)
- **Accuracy**: Trained on 2,000+ annotated fire images

## 📁 Project Structure

```
├── detect_fire.py          # Real-time webcam detection
├── detect_fire_image.py    # Static image detection
├── train_fire.py           # Model training script
├── download_dataset.py     # Dataset download utility
├── Fire_Detection_Colab.ipynb  # Google Colab training notebook
├── requirements.txt        # Python dependencies
└── README.md
```

## 🎥 How It Works

1. **Capture**: Video frames from webcam or load static images
2. **Detect**: YOLOv8 model processes each frame to identify fire/smoke
3. **Alert**: When detected, draws bounding boxes and triggers audio alarm
4. **Track**: Monitors continuously until manually stopped

## 🧠 Model Training

The system comes with pre-trained weights, but you can train on your own data:

- Download fire/smoke datasets from Roboflow Universe
- Fine-tune YOLOv8 on custom datasets
- Use Google Colab for free GPU training (5 epochs ~3-5 minutes)

## 🔧 Configuration

Edit `data.yaml` to customize:
- Dataset paths
- Class names
- Training parameters

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 👨‍💻 Author

**Aksh Patel**  
GitHub: [@Aksh1810](https://github.com/Aksh1810)

---

⭐ **Star this repo if you find it useful!**
