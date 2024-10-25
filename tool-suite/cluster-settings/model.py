import glob
from PIL import Image
from ultralytics import YOLO  
import json
import hashlib
from os import path

class Box:
    def __init__(self, cls: int | float, conf : float, data: list[float], xywh: list[float]):
        self.cls = int(cls)
        self.conf = conf
        self.data = data
        self.xywh = xywh

class ClusterSettingModel:
    def __init__(self, model = "yolo11n.pt"):
        self.model_name = model
        self.model = YOLO(model)

    def get_image_files(self, directory: str) -> list[str]:
        files = glob.glob(directory + "/**/*", recursive=True)
        images = []
        for file in files:
            try:
                with Image.open(file) as f:
                    f.verify()
                images.append(file)
            except:
                pass

        return images

    def find_objects(self, img_path: str, save_path: str | None = None) -> dict[str, str | list]:
        with open(img_path, 'rb') as f:
            sha_hash = hashlib.sha256(f.read()).hexdigest()
        
        result = self.model(img_path)
        result = result[0].boxes

        boxes = []
        for box in zip(result.cls, result.conf, result.data, result.xywh):
            no_tensor_data = []
            for data in box:
                no_tensor_data.append(data.cpu().numpy().tolist())
            boxes.append(Box(*no_tensor_data).__dict__)
        
        json_data = {
            "file": path.abspath(img_path),
            "hash": sha_hash,
            "model": self.model_name,
            "boxes": boxes 
        }

        if(save_path):
            with open(save_path, 'w') as f:
                json.dump(json_data, f)

        return json_data
    

    def batch_find_objects(self, img_dir: str, save_path: str | None = None) -> None:
        images = self.get_image_files(img_dir)
        json_data = {
            "directory": path.abspath(img_dir),
            "images": [self.find_objects(img) for img in images]
        }
        
        if(save_path):
            with open(save_path, 'w') as f:
                json.dump(json_data, f)