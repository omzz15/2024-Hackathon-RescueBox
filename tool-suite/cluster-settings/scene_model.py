from ultralytics import YOLO
import os

class ClusterSettingModel:
    def __init__(self):
        self.model = YOLO("yolo11n.yaml")
    
    def train(self):
        directory = "C:/Users/15084/Desktop/RescueLabHackathon/2024-Hackathon-RescueBox/tool-suite/cluster-settings/indoorCVPR_09/Images"
        for folder in os.listdir(directory):
            for file in os.listdir(os.path.join(directory, folder)):
                print(os.path.join(directory, folder, file))
                
Yolo = ClusterSettingModel()
Yolo.train()