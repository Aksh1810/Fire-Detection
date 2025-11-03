import cv2
import numpy as np
import time
import subprocess

def play_alarm(path="alarm.wav"):
    try:
        subprocess.Popen(["afplay", path])
    except Exception as e:
        print("Alarm error:", e)

FIRE_PIXEL_THRESHOLD = 12000 
MIN_CONTOUR_AREA = 800        
FLICKER_THRESHOLD = 0.08      
FLICKER_COUNT_WINDOW = 10     
ALARM_COOLDOWN = 8            


cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

prev_mask = None
flicker_history = []
last_alarm = 0

print("Starting HSV+motion fire detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_fire = np.array([5, 150, 150])
    upper_fire = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    mask = cv2.GaussianBlur(mask, (7,7), 0)

    fire_pixels = cv2.countNonZero(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contour_found = False
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
            large_contour_found = True

    if prev_mask is None:
        prev_mask = mask.copy().astype("float")
        flicker_fraction = 0.0
    else:
        diff = cv2.absdiff(mask, cv2.convertScaleAbs(prev_mask))
        diff_pixels = cv2.countNonZero(diff)
        flicker_fraction = diff_pixels / max(1, fire_pixels)
        prev_mask = cv2.addWeighted(prev_mask, 0.9, mask.astype("float"), 0.1, 0)

    flicker_history.append(flicker_fraction)
    if len(flicker_history) > FLICKER_COUNT_WINDOW:
        flicker_history.pop(0)
    avg_flicker = sum(flicker_history) / (len(flicker_history) + 1e-9)

    detection = (fire_pixels > FIRE_PIXEL_THRESHOLD) and large_contour_found and (avg_flicker > FLICKER_THRESHOLD)

    if detection:
        try:
            cv2.putText(frame, "FIRE DETECTED!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            now = time.time()
            if now - last_alarm > ALARM_COOLDOWN:
                play_alarm("alarm.wav")
                last_alarm = now
        except Exception as e:
            print("Detection/alarm error:", e)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()