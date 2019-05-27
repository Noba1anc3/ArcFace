#coding=utf-8
from PIL import Image, ImageFont, ImageDraw
from config import get_config
from Learner import face_learner
from utils import load_facebank, prepare_facebank_face

import os
import cv2
import time
import numpy

class FaceRecognitionInference(object):
    def __init__(self, update):
        self.conf = get_config(False)
        self.learner = face_learner(self.conf, True)
        self.learner.load_state(self.conf, 'ir_se50.pth', True, True)
        self.learner.model.eval()
        self.learner.threshold = 0.8
        print('learner loaded')

        if update:
            self.targets, self.names = prepare_facebank_face(self.conf, self.learner.model, face_detect)
            print('facebank updated')
        else:
            self.targets, self.names = load_facebank(self.conf)
            print('facebank loaded')

    def infer(self, faces):
        """
        :param image: the input image
        :param face:  the field of the face(aligned) in image
        :return: the recognized name for the input face
        """
        results, score = self.learner.infer(self.conf, faces, self.targets, tta=True)
        return results, score

if __name__ == '__main__':

    SHOW = True
    test_folder = "./test_images/"

    from face_detector.face_detector_inference import FaceDetectorInference
    import face_detector.config as face_detector_config

    face_detect = FaceDetectorInference(face_detector_config, landmark_flag=True, aligned_flag=True)
    face_recog = FaceRecognitionInference(update=True)

    for image_name in os.listdir(test_folder):
        image = cv2.imread(os.path.join(test_folder, image_name))
        start_time = time.time()
        boxes, faces, landmarks = face_detect.infer(image)
        # 如果有人脸才进行识别
        if len(boxes) != 0:
            print("detect time: ", time.time() - start_time)
            start_time = time.time()
            results, scores = face_recog.infer(faces)
            print("recog time: ", time.time() - start_time)
            print("="*100)

            if SHOW:
                for result, box in zip(results, boxes):
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)

                    cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pilimg = Image.fromarray(cv2img)

                    draw = ImageDraw.Draw(pilimg)
                    font = ImageFont.truetype("wqy-microhei.ttc", 40, encoding="utf-8")
                    draw.text((box[0], box[1]), face_recog.names[result], (0, 255, 0), font=font)
                    image = cv2.cvtColor(numpy.array(pilimg), cv2.COLOR_RGB2BGR)

                cv2.imshow("image", image)
                cv2.waitKey(0)