# Fire Detection System using YOLOv8

This project implements a real-time fire detection system using Python and the YOLOv8 AI model.

## Prerequisites

1.  Python 3.8 or higher.
2.  A webcam (for real-time detection).

## Installation

1.  Clone this repository (if you haven't already).
2.  Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Detection

To start the fire detection system, run:

```bash
python detect_fire.py
```

**Note:** By default, the script looks for a model named `fire_model.pt`. If it doesn't find it, it will use the standard `yolov8n.pt` model, which detects common objects (people, cars, etc.) but **NOT** fire specifically.

To detect fire, you need a custom-trained model.

### 2. Training a Custom Model

To detect fire accurately, you need to train the model on a fire dataset.

#### Step A: Get a Dataset
Since automated downloading can be tricky with permissions, the best way is to get a dataset directly from Roboflow Universe.

1.  **Find a Dataset**:
    *   Go to [Roboflow Universe - Fire Datasets](https://universe.roboflow.com/search?q=fire).
    *   Choose a dataset (e.g., "Fire Detection").
2.  **Download**:
    *   Click "Download Dataset".
    *   Select **YOLOv8** format.
    *   Select "Show Download Code".
    *   Copy the Python code snippet.
3.  **Run the Download**:
    *   Paste the copied code into a new file (e.g., `download_manual.py`) or just run it in your python console.
    *   Alternatively, use the `download_dataset.py` script if you know the workspace and project names:
        ```bash
        # Edit download_dataset.py to use the specific workspace/project you found
        ```
    *   Once downloaded, note the folder name.

#### Step B: Train the Model
Once the dataset is downloaded, the script will tell you where it is (usually in a folder named `fire-detection-...`).

1.  **Update Configuration**:
    *   Open `train_fire.py`.
    *   Update the `data='data.yaml'` line to point to the `data.yaml` file inside the downloaded dataset folder.
        *   Example: `data='fire-detection-1/data.yaml'`

2.  **Run Training**:
    ```bash
    python train_fire.py
    ```

After training, the best model weights will be saved in `runs/detect/train/weights/best.pt`. Rename this file to `fire_model.pt` and place it in the main directory to use it with `detect_fire.py`.

## Project Structure

*   `detect_fire.py`: Main script for real-time detection.
*   `train_fire.py`: Script to train a new model.
*   `requirements.txt`: List of dependencies.
*   `data.yaml`: Configuration file for training (template).
