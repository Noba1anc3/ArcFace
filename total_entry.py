
from face_detector_solver import FaceDetectorSolver
from face_recognition_Solver import FaceRecognitionSolver
from emotion_recognition_solver import EmotionRecognitionSolver
import multiprocessing

class TotalEntry(object):
    def __init__(self):
        self.face_detector = FaceDetectorSolver()
        self.face_recognition = FaceRecognitionSolver()
        self.emotion_recognition = EmotionRecognitionSolver()

    def run(self, image):
        dets, landmarks = self.face_detector.run(image)

        pool = multiprocessing.Pool(processes=2)
        pool.apply_async(self.face_recognition.run(image, dets))
        pool.apply_async(self.emotion_recognition.run(image, dets, landmarks))

        pool.close()
        pool.join()