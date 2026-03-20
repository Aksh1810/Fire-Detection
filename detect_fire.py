import cv2
from ultralytics import YOLO
import math
import os
import shlex
import argparse


def get_alert_audio_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        'ma-ka-bhosda.mp3',
        'ma-ka-bhosda-aag.mp3',
    ]
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path
    return None


def play_alert_audio(audio_path):
    if not audio_path:
        os.system('afplay /System/Library/Sounds/Sosumi.aiff &')
        return
    os.system(f"afplay {shlex.quote(audio_path)} &")

def main():
    model_path = 'fire_model.pt'
    alert_audio_path = get_alert_audio_path()
    
    if not os.path.exists(model_path):
        print(f"Warning: '{model_path}' not found. Using standard 'yolov8n.pt'.")
        model = YOLO('yolov8n.pt')
    else:
        print(f"Loading custom model: {model_path}")
        model = YOLO(model_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='Source: "0" for webcam or path to video file')
    args = parser.parse_args()

    if args.source == '0':
        print("Attempting to open webcam 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam 0. Trying webcam 1...")
            cap = cv2.VideoCapture(1)
    else:
        print(f"Opening video file: {args.source}")
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"Error: Could not open source '{args.source}'.")
        return

    cap.set(3, 1280)
    cap.set(4, 720)

    if alert_audio_path:
        print(f"Using alert audio: {alert_audio_path}")
    else:
        print("Custom alert MP3 not found. Falling back to macOS default sound.")

    fire_detected_last_frame = False

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        fire_detected_current_frame = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                current_class = model.names[cls]
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                label = f'{current_class} {conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                if "fire" in current_class.lower() or "smoke" in current_class.lower():
                    fire_detected_current_frame = True
                    print("!!! FIRE DETECTED !!!")

        if fire_detected_current_frame and not fire_detected_last_frame:
            play_alert_audio(alert_audio_path)

        fire_detected_last_frame = fire_detected_current_frame

        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

