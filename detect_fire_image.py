import cv2
from ultralytics import YOLO
import math
import os
import sys

def detect_fire_in_image(image_path, model_path='fire_model.pt'):
    # Load the model
    if not os.path.exists(model_path):
        print(f"Warning: '{model_path}' not found. Using standard 'yolov8n.pt'.")
        print("Note: The standard model detects common objects, not specifically fire.")
        model = YOLO('yolov8n.pt')
    else:
        print(f"Loading custom model: {model_path}")
        model = YOLO(model_path)

    # Read the image
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image '{image_path}'.")
        return

    print(f"Processing image: {image_path}")
    
    # Run inference
    results = model(img)

    # Process results
    fire_detected = False
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
            current_class = model.names[cls]
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Label
            label = f'{current_class} {conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

            # Check for fire/smoke
            if "fire" in current_class.lower() or "smoke" in current_class.lower():
                fire_detected = True
                print(f"!!! {current_class.upper()} DETECTED with {conf*100:.1f}% confidence !!!")

    # Save result
    output_path = image_path.rsplit('.', 1)[0] + '_detected.' + image_path.rsplit('.', 1)[1]
    cv2.imwrite(output_path, img)
    print(f"Result saved to: {output_path}")
    
    # Display result
    cv2.imshow("Fire Detection Result", img)
    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if fire_detected:
        print("\n✅ Fire or smoke was detected in the image!")
    else:
        print("\n❌ No fire or smoke detected in the image.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_fire_image.py <image_path>")
        print("Example: python detect_fire_image.py test_fire.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    detect_fire_in_image(image_path)
