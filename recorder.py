import cv2
import numpy as np
from os.path import join
from datetime import datetime

class Recorder():
    def __init__(self, fps=15, height=480, width=640, folder = 'videos') -> None:
        self.cameraCapture = cv2.VideoCapture(0)
        self.height = height
        self.width = width
        self.fps = float(fps)
        self.folder = folder

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video = cv2.VideoWriter(join(self.folder, f'pacman_{datetime.today().strftime("%Y_%m_%d_%H_%M_%S")}.mp4'), self.fourcc, self.fps, (self.width, self.height))

    def add_image(self, img) -> None:
        nparr = np.frombuffer(img, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.resize(img_np, (self.width, self.height), cv2.INTER_LANCZOS4)
        self.video.write(frame)

    def close_recording(self) -> None:
        self.video.release()
        cv2.destroyAllWindows()