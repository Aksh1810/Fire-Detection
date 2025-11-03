Fire Detection Using HSV + Motion

Overview

This project implements a real-time fire detection system using a computer’s webcam. It combines color-based detection (using HSV color filtering) and motion/flicker analysis to identify regions likely to contain fire. When fire is detected, the system highlights the affected areas in the video feed and triggers an alarm sound.

The detection method is designed to minimize false positives by combining three checks:
	1.	Number of fire-colored pixels in the frame.
	2.	Presence of sufficiently large fire-like contours.
	3.	Temporal flicker/motion in the fire-colored regions.

Features
	•	Real-time fire detection using a standard webcam.
	•	Visual highlighting of detected fire areas.
	•	Audible alarm when fire is detected.
	•	Adjustable sensitivity parameters for different environments.
	•	Simple interface with live video feed and mask preview.

Requirements
	•	Python 3.11 or higher
	•	OpenCV
	•	NumPy
	•	SimpleAudio or other Python audio library
	•	Webcam accessible to the system

Usage
	1.	Activate the project’s Python virtual environment.
	2.	Ensure the alarm sound file is present in the project folder.
	3.	Run the fire detection script.
	4.	Press q to exit the application.

Notes
	•	Detection accuracy depends on lighting conditions, camera quality, and parameter tuning.
	•	Designed and tested primarily for macOS, but can be adapted to other platforms.
	•	Not intended as a replacement for professional fire safety systems; it’s a demonstration and research tool.