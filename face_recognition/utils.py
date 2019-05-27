#coding=utf-8

import os
import io
import cv2
import time
import torch

import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms as trans
from PIL import ImageDraw, ImageFont, Image
from datetime import datetime
from pathlib import Path

from model import l2_norm
from data.data_pipe import de_preprocess
plt.switch_backend('agg')

def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn

def inference(conf, args, targets, names, learner, face_detecter):
    num_of_frame = 0
    learner.threshold = args.threshold
    cap = cv2.VideoCapture(str(conf.facebank_path/args.file_name))
    cap.set(cv2.CAP_PROP_POS_MSEC, 0)
    video_fps = '%.1f'%(cap.get(cv2.CAP_PROP_FPS))
    begin = time.time()
    if args.save:
        video_writer = cv2.VideoWriter(str(conf.facebank_path/'{}'.format(args.save_name)),cv2.VideoWriter_fourcc(*'XVID'), int(video_fps), (1920,1080))
    while cap.isOpened():
        isSuccess,frame = cap.read()
        if isSuccess:
            start = time.time()
            image = Image.fromarray(frame[...,::-1])
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)  # PIL TO NUMPY

            if num_of_frame % args.gap == 0:
                try:
                    bboxes, faces, landmarks = face_detecter.infer(image)
                except:
                    bboxes = []
                    faces = []
                if len(bboxes) != 0:
                    results, score = learner.infer(conf, faces, targets, tta=True)

            if len(bboxes) != 0:
                for result, bbox in zip(results, bboxes):
                    if args.score:
                        frame = draw_box_name(bbox, names[result] + " score = " + str(score[0])[7:str(score[0]).index(",",7)], frame)
                    else:
                        frame = draw_box_name(bbox, names[result], frame)

            num_of_frame += 1
            now = time.time()
            second_video = '%.1f'%(num_of_frame/25)         #time of video since start
            second_real = '%.1f'%(now - begin)              #time of algorithm since start
            fps = '%.0f'%(1/(now - start))                  #current frame per second
            avg_fps = '%.0f'%(num_of_frame/(now - begin))   #average frame per second on processing the video
            speed_rate = 100
            if float(second_real) != 0:
                speed_rate = '%.0f'%(100*float(second_video)/float(second_real))

            fps = 'VideoFPS: ' + video_fps + '  AvgFPS:' + avg_fps + '  FPS:' + fps + '  ' + second_video + 's' + '  Speed:' + str(speed_rate) + '%'
            frame = draw_fps(fps, frame)

            cv2.imshow('Out Stream',frame)

            if num_of_frame % 25 == 0:
                print('Video Time: ' + second_video + 's' + '  ' + 'Real Time: ' + second_real + 's')

            if args.save:
                video_writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    if args.save:
        video_writer.release()

def load_facebank(conf):
    summary = ''
    embeddings = torch.load(conf.facebank_path/'facebank.pth')
    names = np.load(conf.facebank_path/'names.npy')

    fileinfo = os.stat('./data/facebank/names.npy')
    timeStamp = fileinfo.st_mtime
    date = str(datetime.fromtimestamp(timeStamp))[:19]

    for name in names:
        summary += name + ' '
    summary = summary[8:-1]

    print('***********************************Facebank Loaded**********************************')
    print('****************************Version: ' + date + '****************************')
    print('人脸库成员: ' + summary)
    print('')
    return embeddings, names

def prepare_facebank_face(conf, model, face_detector, tta=True):
    """ Use face_detector to detect all the faces in the face-database,
        then save facebank and names file to the directory.
        Arguments:
            conf: configuration of face_recognition.
            model: face_recognition model
            face_detector: face_detector inference
        Returns:
            embeddings: all the face embeddings of all the people, size = [names,512]
            names: all the name of people in the facebank, size = [1,num of people]
    """
    total = 0
    unlock = 1
    model.eval()

    success = []
    embeddings = []
    names = ['Unknown']

    for path in conf.facebank_path.iterdir():
        if path.is_file():
            continue
        else:
            embs = []
            success_num = 0
            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    file = str(file)
                    img = cv2.imread(file)

                    if unlock:
                        try:
                            boxes, faces, _ = face_detector.infer(img)
                        except:
                            unlock = 0
                            print("********************************Facebank Initialized********************************")
                        unlock = 0

                    try:
                        boxes, faces, _ = face_detector.infer(img)
                    except Exception as e:
                        print("未能采集到人脸, 注册 " + file[13:] + " 时失败")
                        print(e)
                        continue

                    face = cv2.resize(faces[0], (112, 112))
                    img = Image.fromarray(np.uint8(face))

                    with torch.no_grad():
                        if tta:
                            mirror = trans.functional.hflip(img)
                            emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                            emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                            embs.append(l2_norm(emb + emb_mirror))
                        else:
                            embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))

                    success_num += 1
                    print("成功注册 " + file[13:])

        if len(embs) == 0:
            continue

        embedding = torch.cat(embs).mean(0, keepdim=True)
        embeddings.append(embedding)
        names.append(path.name)
        success.append(success_num)

    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, conf.facebank_path / 'facebank.pth')
    np.save(conf.facebank_path / 'names', names)

    for i in range(len(success)):
        total += success[i]
        print("成功注册 " + str(names[i+1]) + " 的照片共计 " + str(success[i]) + "张")

    print(str(len(names) - 1) + " 位同学共计 " + str(total)+ " 张照片已成功被注册到系统当中")

    print('**********************************Facebank Updated**********************************')
    fileinfo = os.stat('./data/facebank/names.npy')
    timeStamp = fileinfo.st_mtime
    date = str(datetime.fromtimestamp(timeStamp))[:19]

    summary = ''
    for name in names:
        summary += name + ' '
    summary = summary[8:-1]
    print('****************************Version: ' + date + '****************************')
    print('人脸库成员: ' + summary)
    print('')
    return embeddings, names

def update_facebank_single(conf, model, face_detector, face_filename, tta=True):
    """ Use face_detector to detect all the faces in the face-database,
        then save facebank and names file to the directory.
        Arguments:
            conf: configuration of face_recognition.
            model: face_recognition model
            face_detector: face_detector inference
            face_filename: str of username_count.jpg
        Returns:
            embeddings: all the face embeddings of all the people, size = [names,512]
            names: all the name of people in the facebank, size = [1,num of people]
            console: adding information of explicit photo
    """

    unlock = 1
    model.eval()

    success = 0
    embeddings = []
    names = ['Unknown']

    for path in conf.facebank_path.iterdir():
        if path.is_file():
            continue
        else:
            embs = []
            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    file = str(file)
                    img = cv2.imread(file)

                    if unlock:
                        try:
                            boxes, faces, _ = face_detector.infer(img)
                        except:
                            unlock = 0
                            print("********************************Facebank Initialized********************************")
                        unlock = 0

                    try:
                        boxes, faces, _ = face_detector.infer(img)
                    except Exception as e:
                        print("未能采集到人脸, 注册 " + file[13:] + " 时失败")
                        print(e)
                        continue

                    face = cv2.resize(faces[0], (112, 112))
                    img = Image.fromarray(np.uint8(face))

                    with torch.no_grad():
                        if tta:
                            mirror = trans.functional.hflip(img)
                            emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                            emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                            embs.append(l2_norm(emb + emb_mirror))
                        else:
                            embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
                    if str(path)[14:] == face_filename[0:face_filename.index("_",0)]:
                        success += 1
                        if file.find(face_filename) > 0:
                            print("成功注册 " + file[13:])
        if len(embs) == 0:
            continue

        embedding = torch.cat(embs).mean(0, keepdim=True)
        embeddings.append(embedding)
        names.append(path.name)

    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, conf.facebank_path / 'facebank.pth')
    np.save(conf.facebank_path / 'names', names)
    print("更新注册 " + face_filename[0:face_filename.index("_",0)] + " 同学的照片共计 " + str(success) + "张")
    print('**********************************Facebank Updated**********************************')
    return embeddings, names

def update_facebank_multiple(conf, model, face_detector, face_filename, tta=True):
    """ Use face_detector to detect all the faces in the face-database,
        then save facebank and names file to the directory.
        Arguments:
            conf: configuration of face_recognition.
            model: face_recognition model
            face_detector: face_detector inference
            face_filename: array of strs of username_count.jpg
        Returns:
            embeddings: all the face embeddings of all the people, size = [names,512]
            names: all the name of people in the facebank, size = [1,num of people]
            console: adding information of the array of photo
    """

    unlock = 1
    model.eval()

    success = 0
    embeddings = []
    names = ['Unknown']

    for path in conf.facebank_path.iterdir():
        if path.is_file():
            continue
        else:
            embs = []
            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    file = str(file)
                    img = cv2.imread(file)

                    if unlock:
                        try:
                            boxes, faces, _ = face_detector.infer(img)
                        except:
                            unlock = 0
                            print("********************************Facebank Initialized********************************")
                        unlock = 0

                    try:
                        boxes, faces, _ = face_detector.infer(img)
                    except Exception as e:
                        print("未能采集到人脸, 注册 " + file[13:] + " 时失败")
                        print(e)
                        continue

                    face = cv2.resize(faces[0], (112, 112))
                    img = Image.fromarray(np.uint8(face))

                    with torch.no_grad():
                        if tta:
                            mirror = trans.functional.hflip(img)
                            emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                            emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                            embs.append(l2_norm(emb + emb_mirror))
                        else:
                            embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
                    if str(path)[14:] == face_filename[0][0:face_filename[0].index("_",0)]:
                        success += 1
                        for fname in face_filename:
                            if file.find(fname) > 0:
                                print("成功注册 " + file[13:])
        if len(embs) == 0:
            continue

        embedding = torch.cat(embs).mean(0, keepdim=True)
        embeddings.append(embedding)
        names.append(path.name)

    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, conf.facebank_path / 'facebank.pth')
    np.save(conf.facebank_path / 'names', names)
    print("更新注册 " + face_filename[0][0:face_filename[0].index("_",0)] + " 同学的照片共计 " + str(success) + "张")
    print('**********************************Facebank Updated**********************************')
    return embeddings, names

def add_user(conf, username):
    """ Add new user to the face-database
        Arguments:
            conf: configuration of face_recognition.
            username: string of new user's name
    """
    if not (conf.facebank_path / Path(username)).exists():
        os.makedirs(conf.facebank_path / Path(username))
        print(username + " 的人脸库创建成功")
    else:
        print("增加新用户失败, " + username + " 的人脸库已经存在")

def add_pic_over_camera(conf, model, face_detector, name):
    """ Take a pic from camera, and then add the pic to its user's folder path
        if usr's folder doesn't exists, create a folder in the facebank for the usr
        Then, update the facebank and return new targets and names
        Arguments:
            conf: configuration of face_recognition.
            model: face_recognition model
            face_detector: face_detector inference
            name: usr's name
        Returns:
            targets: all the face embeddings of all the people, size = [names,512]
            names: all the name of people in the facebank, size = [1,num of people]
    """
    count = 1
    save_path = conf.facebank_path / Path(name)
    if not save_path.exists():
        save_path.mkdir()
        print(name + " 的人脸库创建成功")

    for _ in save_path.iterdir():
        count += 1
    face_path = str(save_path / Path(name + "_" + str(count) + ".jpg"))
    face_filename = str(Path(name + "_" + str(count) + ".jpg"))

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    while cap.isOpened():
        isSuccess, frame = cap.read()

        if isSuccess:
            frame_text = cv2.putText(frame, 'Press T to take a picture', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Camera", frame_text)

        if cv2.waitKey(1) & 0xFF == ord('t'):
            photo = Image.fromarray(frame[..., ::-1])
            photo = np.array(photo)[..., ::-1]
            cv2.imwrite(face_path, photo)
            print(name + " 同学的照片已添加至人脸库, 路径为：" + face_path)
            #cv2.imwrite(str(save_path / '{}.jpg'.format(str(datetime.now())[:-7].replace(":", "-").replace(" ", "-"))),warped_face)
            break

    cap.release()
    #cv2.destoryAllWindows()
    update_facebank_single(conf, model, face_detector, face_filename)
    targets, names = load_facebank(conf)

    return targets, names

def add_face_single(conf, model, face_detector, username, photo_path):
    """ add new photo to the face-database, if the folder of user do not exists
        it will automatically establish a new folder and put the photo into it
        new photo will be automatically set a name based on num of photos in the folder
        then, update the facebank and return new targets and names
        Arguments:
            conf: configuration of face_recognition.
            model: face_recognition model
            face_detector: face_detector inference
            username: string of user's name of explicit photo
            photo_path: string of saving path of explicit photo
         Returns:
            targets: all the face embeddings of all the people, size = [names,512]
            names: all the name of people in the facebank, size = [1,num of people]
    """

    count = 1
    face = cv2.imread(photo_path)
    path = conf.facebank_path / Path(username)
    if not path.exists():
        os.makedirs(path)
        print(username + " 的人脸库创建成功")
    for _ in path.iterdir():
        count += 1
    face_path = str(path / Path(username + "_" + str(count) + ".jpg"))
    face_filename = str(Path(username + "_" + str(count) + ".jpg"))
    cv2.imwrite(face_path, face)
    print(username + " 同学的照片已添加至人脸库, 路径为：" + face_path)
    update_facebank_single(conf, model, face_detector, face_filename)
    targets, names = load_facebank(conf)
    return targets, names

def add_face_multiple(conf, model, face_detector, username, folder_path):
    """ add a folder of photos to the face-database, if the folder of user do not exists
        it will automatically establish a new folder and put the photos into it
        new photo will be automatically set a name based on num of photos in the folder
        then, update the facebank and return new targets and names
        Arguments:
            conf: configuration of face_recognition.
            model: face_recognition model
            face_detector: face_detector inference
            username: string of user's name of explicit photo
            folder_path: string of saving path of explicit folder
        Returns:
            targets: all the face embeddings of all the people, size = [names,512]
            names: all the name of people in the facebank, size = [1,num of people]
    """

    count = 1
    file_num = 0
    face_filename = []

    for file in Path(folder_path).iterdir():
        if not file.is_file():
            continue
        file_num += 1

    if(file_num != 0):
        path = conf.facebank_path / Path(username)
        if not path.exists():
            os.makedirs(path)
            print(username + " 的人脸库创建成功")
        for _ in path.iterdir():
            count += 1

        for file in Path(folder_path).iterdir():
            if not file.is_file():
                continue
            else:
                face = cv2.imread(str(file))
                face_path = str(path / Path(username + "_" + str(count) + ".jpg"))
                cv2.imwrite(face_path, face)
                face_filename.append(str(Path(username + "_" + str(count) + ".jpg")))
                print(username + " 同学的照片已添加至人脸库, 路径为：" + face_path)
                count += 1
        update_facebank_multiple(conf, model, face_detector, face_filename)
        targets, names = load_facebank(conf)
        return targets, names
    else:
        print(folder_path + " 文件夹下没有图片")

def del_face(conf, model, face_detector, username, fileID):
    """ delete specific user's photo in its folder in the face-database
        then, update the facebank and return new targets and names
        Arguments:
            conf: configuration of face_recognition.
            model: face_recognition model
            face_detector: face_detector inference
            username: string of user's name of explicit photo
            fileID: explicit ID number of the photo
        Returns:
            targets: all the face embeddings of all the people, size = [names,512]
            names: all the name of people in the facebank, size = [1,num of people]
    """
    folder_path = conf.facebank_path / Path(username)
    file_path = conf.facebank_path / Path(username) / Path(username + "_" + str(fileID) + ".jpg")
    if folder_path.exists():
        if(file_path.exists()):
            os.remove(file_path)  # 删除文件
            print("成功删除 " + username + " 人脸库中的文件 " + username + "_" + str(fileID) + ".jpg")
            prepare_facebank_face(conf, model, face_detector)
            targets, names = load_facebank(conf)
            return targets, names
        else:
            print("未在 " + username + " 的人脸库中检测到文件 " + username + "_" + str(fileID) + ".jpg")
    else:
        print("未在系统中检测到 " + username + " 的人脸库")

def del_user(conf, model, face_detector, username):
    """ delete specific user's folder in the face-database
        then, update the facebank and return new targets and names
        Arguments:
            conf: configuration of face_recognition.
            model: face_recognition model
            face_detector: face_detector inference
            username: string of user's name of explicit folder
        Returns:
            targets: all the face embeddings of all the people, size = [names,512]
            names: all the name of people in the facebank, size = [1,num of people]
    """
    import shutil
    path = conf.facebank_path / Path(username)
    if path.exists():
        shutil.rmtree(str(path))
        print("成功删除 " + username + " 的人脸库")
        prepare_facebank_face(conf, model, face_detector)
        targets, names = load_facebank(conf)
        return targets, names
    else:
        print("未在系统中检测到 " + username + " 的人脸库")

def list_user(conf):
    """ list all the user's name in the face-database
        Arguments:
            conf: configuration of face_recognition.
    """
    user = []
    path = conf.facebank_path
    for folder in path.iterdir():
        if not folder.is_file():
            user.append(str(folder)[14:])
    print(user)

def name_normalize(conf):
    """ rename all the photo in the face_database to the format: username_number.jpg
        Arguments:
            conf: configuration of face_recognition.
    """
    for path in conf.facebank_path.iterdir():
        if path.is_file():
            continue
        else:
            count = 1
            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    file_path = str(path) + "/" + str(path)[14:] + "__" + str(count) + ".jpg"
                    os.rename(file, file_path)
                    count += 1

    for path in conf.facebank_path.iterdir():
        if path.is_file():
            continue
        else:
            count = 1
            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    file_path = str(path) + "/" + str(path)[14:] + "_" + str(count) + ".jpg"
                    os.rename(file, file_path)
                    count += 1
    print("Name Normalized")

def draw_box_name(bbox,name,frame):
    cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)

    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype("wqy-microhei.ttc", 15, encoding="utf-8") #/usr/share/fonts/truetype   fc-list :lang=zh
    draw.text((bbox[0],bbox[1]), name, (0,255,0), font=font)
    frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),2)

    return frame

def draw_fps(fps,frame):
    cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)

    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype("wqy-microhei.ttc", 20)
    draw.text((0,0), fps, (0,0,255), font=font)
    frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

    return frame

def get_time():
    return (str(datetime.now())[:-7]).replace(' ','-').replace(':','-')

def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf

hflip = trans.Compose([
            de_preprocess,
            trans.ToPILImage(),
            trans.functional.hflip,
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

def face_reader(conf, conn, flag, boxes_arr, result_arr, learner, mtcnn, targets, tta):
    while True:
        try:
            image = conn.recv()
        except:
            continue
        try:            
            bboxes, faces = mtcnn.align_multi(image, limit=conf.face_limit)
        except:
            bboxes = []
            
        results = learner.infer(conf, faces, targets, tta)
        
        if len(bboxes) > 0:
            print('bboxes in reader : {}'.format(bboxes))
            bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] # personal choice            
            assert bboxes.shape[0] == results.shape[0],'bbox and faces number not same'
            bboxes = bboxes.reshape([-1])
            for i in range(len(boxes_arr)):
                if i < len(bboxes):
                    boxes_arr[i] = bboxes[i]
                else:
                    boxes_arr[i] = 0 
            for i in range(len(result_arr)):
                if i < len(results):
                    result_arr[i] = results[i]
                else:
                    result_arr[i] = -1 
        else:
            for i in range(len(boxes_arr)):
                boxes_arr[i] = 0 # by default,it's all 0
            for i in range(len(result_arr)):
                result_arr[i] = -1 # by default,it's all -1
        print('boxes_arr ： {}'.format(boxes_arr[:4]))
        print('result_arr ： {}'.format(result_arr[:4]))
        flag.value = 0

def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs

# def prepare_facebank(conf, model, mtcnn, tta = True):
#     model.eval()
#     embeddings =  []
#     names = ['Unknown']
#     for path in conf.facebank_path.iterdir():
#         if path.is_file():
#             continue
#         else:
#             embs = []
#             for file in path.iterdir():
#                 if not file.is_file():
#                     continue
#                 else:
#                     try:
#                         img = Image.open(file)
#                         if img.size != (112, 112):
#                             img = mtcnn.align(img)
#                     except:
#                         continue
#
#                     with torch.no_grad():
#                         if tta:
#                             mirror = trans.functional.hflip(img)
#                             emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
#                             emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
#                             embs.append(l2_norm(emb + emb_mirror))
#                         else:
#                             embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
#         if len(embs) == 0:
#             continue
#         embedding = torch.cat(embs).mean(0,keepdim=True)
#         embeddings.append(embedding)
#         names.append(path.name)
#     embeddings = torch.cat(embeddings)
#     names = np.array(names)
#     torch.save(embeddings, conf.facebank_path/'facebank.pth')
#     np.save(conf.facebank_path/'names', names)
#     return embeddings, names

# def show_results(img, bounding_boxes, facial_landmarks = []):
#     """Draw bounding boxes and facial landmarks.
#     Arguments:
#         img: an instance of PIL.Image.
#         bounding_boxes: a float numpy array of shape [n, 5].
#         facial_landmarks: a float numpy array of shape [n, 10].
#     Returns:
#         an instance of PIL.Image.
#     """
#     img_copy = img.copy()
#     draw = ImageDraw.Draw(img_copy)
#
#     for b in bounding_boxes:
#         draw.rectangle([
#             (b[0], b[1]), (b[2], b[3])
#         ], outline = 'white')
#
#     for p in facial_landmarks:
#         for i in range(5):
#             draw.ellipse([
#                 (p[i] - 1.0, p[i + 5] - 1.0),
#                 (p[i] + 1.0, p[i + 5] + 1.0)
#             ], outline = 'blue')
#
#     return img_copy
