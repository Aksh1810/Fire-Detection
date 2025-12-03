from roboflow import Roboflow
import os
import sys

def download_dataset(api_key):
    print("Initializing Roboflow...")
    rf = Roboflow(api_key=api_key)
    
    candidates = [
        ("roboflow-100", "fire-smoke-detection"),
        ("yolofire", "fire-detection-yolo"),
        ("joseph-nelson", "fire-detection"),
        ("public-datasets", "fire-smoke-detection"),
        ("ds", "fire-detection-5")
    ]
    
    dataset = None
    for workspace, project_id in candidates:
        for ver in range(1, 6):
            try:
                print(f"Trying dataset: {workspace}/{project_id} v{ver}...")
                project = rf.workspace(workspace).project(project_id)
                version = project.version(ver)
                dataset = version.download("yolov8")
                
                if dataset and dataset.location:
                    if os.path.exists(dataset.location):
                        files = os.listdir(dataset.location)
                        if len(files) == 1 and files[0].endswith('.zip'):
                            zip_path = os.path.join(dataset.location, files[0])
                            if os.path.getsize(zip_path) < 1000:
                                print(f"Download failed (invalid zip) for {workspace}/{project_id} v{ver}")
                                continue
                        
                        print(f"Success! Dataset downloaded to: {dataset.location}")
                        return dataset.location
            except Exception as e:
                continue
            
    if not dataset:
        print("Could not download any of the candidate datasets.")
        return None

    return dataset.location

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = input("Enter your Roboflow API Key: ")
    
    if not api_key:
        print("API Key is required.")
        sys.exit(1)
        
    download_dataset(api_key)
