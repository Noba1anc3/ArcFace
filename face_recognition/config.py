import torch
from pathlib import Path
from torch.nn import CrossEntropyLoss
from easydict import EasyDict as edict
from torchvision import transforms as trans

def get_config(training = True):
    conf = edict()
    conf.data_path = Path('data')
    conf.work_path = Path('work_space/')
    conf.log_path = conf.work_path/'log'
    conf.save_path = conf.work_path/'save'
    conf.model_path = conf.work_path/'models'
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se'
    conf.embedding_size = 512
    conf.input_size = [112,112]
    conf.use_mobilfacenet = True
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])

    conf.data_mode = 'emore'
    conf.emore_folder = conf.data_path/'faces_emore'
    conf.vgg_folder = conf.data_path/'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path/'faces_ms1m_112x112'

    conf.batch_size = 100
#   conf.batch_size = 200 # mobilefacenet

#--------------------Training Config ------------------------
    if training:
        conf.log_path = conf.work_path/'log'
        conf.save_path = conf.work_path/'save'
        conf.lr = 1e-3
        conf.momentum = 0.9
        conf.num_workers = 6
        conf.pin_memory = True
        conf.milestones = [12,15,18]
        conf.ce_loss = CrossEntropyLoss()

        #   conf.weight_decay = 5e-4
        #   conf.num_workers = 4 when batchsize is 200

#--------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path/'facebank'
        conf.threshold = 1.5
        #conf.face_limit = 3
        #conf.min_face_size = 30
        # the larger this value, the faster deduction, comes with tradeoff in small faces

    return conf