import glob
from PIL import Image
from torchvision import transforms
import torch
import numpy
import cv2

class ClusterSettingModel:
    def __init__(self):
        import torch
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        self.model.eval()

    def get_image_files(self, directory: str) -> list[str]:
        # TODO maybe do this without loading all images?
        print(f"Loading images from {directory}...")
        files = glob.glob(directory + "/**/*")
        images = []
        for file in files:
            try:
                with Image.open(file) as f:
                    f.verify()
                images.append(file)
            except:
                print(f"Unable to open {file} as image")

        print(f"Done loading images")

        return images

    def remove_human(self, img_path: str) -> None:
        input_image = Image.open(img_path)
        input_image = input_image.convert("RGB")

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        # # create a color pallette, selecting a color for each class
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

        # # plot the semantic segmentation predictions of 21 classes in each color
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
        r.putpalette(colors)

        open_cv_image = numpy.array(r)
        cv2.imshow("img", open_cv_image)
        cv2.imshow("img2", open_cv_image)


