#coding=utf-8

import utils
import argparse
import face_detector.config as face_detector_config

from Learner import face_learner
from face_recognition.config import get_config
from face_detector.face_detector_inference import FaceDetectorInference

#mobilefacenet 0.5 0.6

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-tta", "--tta", help = "whether test time augmentation", action = "store_true")
    parser.add_argument("-fin", "--file_name", help = "video file name", default = 'video.mp4', type = str)
    parser.add_argument("-fout", "--save_name", help = "output file name", default = 'out_stream.avi', type = str)
    parser.add_argument('-s','--source', help = 'ip means ip_camera, local means video file', default = 'ip', type = str)

    parser.add_argument('-t','--threshold',help = 'threshold to decide identical faces', default = 0.6)
    parser.add_argument("-g", "--gap", help = "how often does the algorithm run", default = 1, type = int)

    parser.add_argument("-sa", "--save", help = "whether save", default = True, action = "store_true")
    parser.add_argument("-sc", "--score", help = "whether show the confidence score", default = False, action = "store_true")
    parser.add_argument("-u", "--update", help = "whether perform update the facebank", default = True, action = "store_true")
    args = parser.parse_args()

    conf = get_config(False)
    utils.name_normalize(conf)

    learner = face_learner(conf, True)
    learner.load_state(conf, 'mobilefacenet.pth', True, True)
    learner.model.eval()
    face_detecter = FaceDetectorInference(face_detector_config, landmark_flag=True, aligned_flag=True)

    if args.update:
        targets, names = utils.prepare_facebank_face(conf, learner.model, face_detecter, tta=True)
    else:
        targets, names = utils.load_facebank(conf)

    utils.inference(conf, args, targets, names, learner, face_detecter)

    #targets, names = utils.add_face_single(conf, learner.model, face_detecter, username = '森', photo_path = 'data/facebank/mimori/mimori_1.jpg')
    #targets, names = utils.add_face_multiple(conf, learner.model, face_detecter, username = 'usrname', folder_path = 'data/folder')
    #targets, names = utils.add_pic_over_camera(conf, learner.model, face_detecter, name = 'usrname')

    #utils.inference(conf, args, targets, names, learner, face_detecter)
