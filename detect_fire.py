import cv2
from ultralytics import YOLO
import math
import os

def main():
    # Load the model
    # If you have a custom trained model for fire, change the path below to 'best.pt' or 'fire.pt'
    model_path = 'fire_model.pt'
    
    if not os.path.exists(model_path):
        print(f"Warning: '{model_path}' not found. Using standard 'yolov8n.pt'.")
        print("Note: The standard model detects common objects (person, car, etc.), not specifically fire.")
        print("To detect fire, you need to train a model or download a pre-trained fire detection model.")
        model = YOLO('yolov8n.pt')
    else:
        print(f"Loading custom model: {model_path}")
        model = YOLO(model_path)

    # specific classes to detect (if using standard model, we can't filter for fire easily)
    # classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    #               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    #               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    #               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    #               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    #               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    #               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    #               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    #               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    #               "teddy bear", "hair drier", "toothbrush"
    #               ]

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='Source: "0" for webcam or path to video file')
    args = parser.parse_args()

    # Start capture
    if args.source == '0':
        print("Attempting to open webcam 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam 0. Trying webcam 1...")
            cap = cv2.VideoCapture(1)
    else:
        print(f"Opening video file: {args.source}")
        cap = cv2.VideoCapture(args.source)
    
    # Check if opened correctly
    if not cap.isOpened():
        print(f"Error: Could not open source '{args.source}'.")
        print("If trying to use a webcam on macOS, ensure Terminal has Camera permissions in System Settings.")
        return

    # Set resolution (optional)
    cap.set(3, 1280)
    cap.set(4, 720)

    # Track if fire was detected in the last frame to avoid repeated alerts
    fire_detected_last_frame = False

    while True:
        success, img = cap.read()
        if not success:
            break

        # Run inference
        results = model(img, stream=True)

        # Track if fire is detected in the current frame
        fire_detected_current_frame = False

        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                
                # Class Name
                cls = int(box.cls[0])
                
                # If using standard model, just show everything. 
                # If using custom fire model, usually class 0 is fire (or whatever the dataset defined).
                
                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Label
                # current_class = classNames[cls] if cls < len(classNames) else str(cls)
                # If using custom model, we might not have class names mapped, so we use the model's names
                current_class = model.names[cls]
                
                label = f'{current_class} {conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, (0, 0, 255), -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                # Check for fire/smoke
                if "fire" in current_class.lower() or "smoke" in current_class.lower():
                    fire_detected_current_frame = True
                    print("!!! FIRE DETECTED !!!")

        # Play alert sound only when fire is FIRST detected (transition from False to True)
        if fire_detected_current_frame and not fire_detected_last_frame:
            # Play system alert sound (non-blocking)
            os.system('afplay /System/Library/Sounds/Sosumi.aiff &')

        # Update the last frame status
        fire_detected_last_frame = fire_detected_current_frame

        cv2.imshow("Image", img)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
