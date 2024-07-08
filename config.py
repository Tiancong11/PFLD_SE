from torchvision import transforms as trans
from easydict import EasyDict as edict
from utils.utils import get_time
import os
import torch


def get_config():
    cfg = edict()
    # SEED（种子）是用于初始化随机数生成器的一个值
    cfg.SEED = 2023
    cfg.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.GPU_ID = 0

    cfg.TRANSFORM = trans.Compose([trans.ToTensor(),
                                   trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    cfg.MODEL_TYPE = 'PFLD'  # [PFLD, PFLD_GhostNet, PFLD_GhostNet_Slim, PFLD_GhostOne]
    cfg.INPUT_SIZE = [112, 112]
    cfg.WIDTH_FACTOR = 1
    cfg.LANDMARK_NUMBER = 98

    cfg.TRAIN_BATCH_SIZE = 32
    cfg.VAL_BATCH_SIZE = 8

    cfg.TRAIN_DATA_PATH = './data/train_data_repeat80/list.txt'
    cfg.VAL_DATA_PATH = './data/test_data_repeat80/list.txt'

    cfg.EPOCHES = 20
    cfg.LR = 1e-4
    cfg.WEIGHT_DECAY = 1e-6
    cfg.NUM_WORKERS = 8
    # 学习率调度器：表示在第 55、65 和 75 个 epoch 时调整学习率
    cfg.MILESTONES = [55, 65, 75]

    # 判断是否需要从已保存的模型中恢复训练。
    # cfg.RESUME = False：这行代码首先将RESUME设置为False，意味着默认情况下不会从已保存的模型中恢复训练。
    cfg.RESUME = False
    if cfg.RESUME:
        cfg.RESUME_MODEL_PATH = ''

    create_time = get_time()
    cfg.MODEL_PATH = './checkpoint/TC/models/{}_{}_{}_{}/'.format(cfg.MODEL_TYPE, cfg.WIDTH_FACTOR, cfg.INPUT_SIZE[0], create_time)
    cfg.LOG_PATH = './checkpoint/TC/log/{}_{}_{}_{}/'.format(cfg.MODEL_TYPE, cfg.WIDTH_FACTOR, cfg.INPUT_SIZE[0], create_time)
    cfg.LOGGER_PATH = os.path.join(cfg.MODEL_PATH, "train.log")
    if not os.path.exists(cfg.MODEL_PATH):
        os.makedirs(cfg.MODEL_PATH)

    return cfg
