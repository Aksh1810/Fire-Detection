from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8n.pt')
    
    print("Starting training...")
    try:
        results = model.train(data='fire-2/data.yaml', epochs=5, imgsz=640)
        print("Training completed successfully!")
        print(f"Best model saved at: {results.save_dir}")
    except Exception as e:
        print(f"Error during training: {e}")
        print("Make sure you have a 'data.yaml' file and your dataset is correctly structured.")

if __name__ == '__main__':
    train_model()

