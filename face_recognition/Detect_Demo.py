from PIL import Image
from mtcnn import MTCNN
from utils import show_results
import os

data_dir = "./detect_demo/"
face_dector = MTCNN()

for index, image_name in enumerate(os.listdir(data_dir)):
    #start_time = time.time()
    img = Image.open(os.path.join(data_dir, image_name))
    bounding_boxes, landmarks = face_dector.detect_faces(img)
    image = show_results(img, bounding_boxes, landmarks)
    image.show()