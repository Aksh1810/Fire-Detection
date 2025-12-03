from roboflow import Roboflow

print("Initializing Roboflow...")
rf = Roboflow(api_key="wCtd6yHELED2FkT0Vrlp")
print("Accessing project...")
project = rf.workspace("gadjiiavov-n4n8k").project("fire-fhsxx")
version = project.version(2)
print("Downloading dataset...")
dataset = version.download("yolov8")
print(f"Dataset downloaded to: {dataset.location}")

