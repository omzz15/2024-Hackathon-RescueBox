from typing import TypedDict
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import *
import glob
import cv2

server = MLServer(__name__)

class ClusterSettingsInputs(TypedDict):
    images_path: DirectoryInput

class ClusterSettingsParameters(TypedDict):
    pass

def create_cluster_settings_task_schema() -> TaskSchema:
    input_schema = InputSchema(
        key="images_path",
        label="Path to images to cluster by setting",
        subtitle="Select a directory to find images and cluster by setting",
        input_type=InputType.DIRECTORY
    )
    return TaskSchema(
        inputs = [input_schema],
        parameters=[]
    )

@server.route(
    "/cluster_settings",
    task_schema_func=create_cluster_settings_task_schema,
    short_title="Cluster settings",
    order=0
)

def transform_case(inputs: ClusterSettingsInputs, parameters: ClusterSettingsParameters) -> ResponseBody:
    


    return ResponseBody(root=TextResponse(value="hello"))

if __name__ == '__main__':
    # Run a debug server
    server.run()
