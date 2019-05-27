
class FaceRecognitionSolver(object):
    """
    face recognition api
    """
    def __init__(self, landmark_flag, aligned_flag):
        self.landmark_flag = landmark_flag
        self.aligned_flag = aligned_flag

    def run(self, image):
        return