from pathlib import Path
from typing import Optional
from cv2 import Mat
import numpy as np

class ClusterSettingModel:
    def __init__(self):
        import inference
        self.model = inference.get_model(model_id="yolov8n-seg-640")

    def get_image_files(self, directory: str) -> dict[Path, Mat]:
        
        # TODO maybe do this without loading all images?
        print(f"Loading images from {directory}...")
        files = glob.glob(directory + "/**/*")
        images = {}
        for file in files:
            try:
                img = cv2.imread(file)
                images[Path(file)] = img
            except:
                print(f"Unable to open {file} as image")

        print(f"Done loading images")

        return images

    def remove_human(self, img : Mat) -> Mat:
        result = self.model.infer(img)[0]
