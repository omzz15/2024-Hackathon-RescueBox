import glob
from pathlib import Path
from typing import Optional
import cv2

class ClusterSettingModel:
    def __init__(self):
        import inference
        self.model = inference.get_model(model_id="yolov8n-seg-640")

    def get_image_files(self, directory: str) -> dict[Path, cv2.Mat]:
        
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

    def _validate_audio_path(self, audio_path: str) -> None:
        if audio_path is None:
            raise ValueError("audio_path cannot be None")

    def transcribe(self, audio_path: str, out_dir: Optional[str] = None) -> str:
        self._validate_audio_path(audio_path)
        res: str = self.model.transcribe(str(audio_path))["text"]  # type: ignore
        if out_dir:
            self._write_res_to_dir([{"file_path": str(audio_path), "result": res}], out_dir)
        return res

    def transcribe_batch(self, audio_paths: list[str]) -> list[dict[str, str]]:
        return [
            {"file_path": str(audio_path), "result": self.transcribe(str(audio_path))}
            for audio_path in audio_paths
        ]

    def _write_res_to_dir(self, res: list[dict[str, str]], out_dir: str) -> None:
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        for r in res:
            with open(out_dir_path / (r["file_path"].split("/")[-1].split(".")[0] + ".txt"), "w") as f:
                f.write(r["result"])

    def transcribe_files_in_directory(
        self, input_dir: str, out_dir: Optional[str] = None
    ) -> list[dict[str, str]]:
        res = self.transcribe_batch([str(file) for file in self.get_audio_files(input_dir)])
        if out_dir:
            self._write_res_to_dir(res, out_dir)
        return res
