import os
import yaml
import platform
import json

import numpy as np
from easydict import EasyDict as edict


config = edict()
config.DATASET = edict()
config.DATASET.DIR = "/netscratch/millerdurai/Datasets/"


config.CHECKPOINT_LOAD_LATEST = True

config.DEBUG = 0
config.BATCH_SIZE = 8
config.N_GPUS = 1
config.PRINT_FREQ = 50


config.POSE_3D = edict()
config.DETECTION = edict()

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_DATA')

config.POSE_3D.CHECKPOINT_PATH = os.path.join(_DATA_DIR, 'model_weights.pth')
config.DETECTION.HAND_ARM_PATH = os.path.join(_DATA_DIR, 'epoch_460.pth')
config.DETECTION.HAND_PATH = os.path.join(_DATA_DIR, 'detector.torchscript')

config.MANO_PATH = os.path.join(_DATA_DIR, "mano")
config.ARM_PCA_PATH = os.path.join(_DATA_DIR, "models/param_data.npy")

config.DATASET.TRAIN_NAME = 'CombinedDataset'
config.DATASET.TEST_NAME = 'ARCTIC'
config.POSE_3D.ARM = True
config.POSE_3D.ROT_6D = True
config.POSE_3D.DECODER_ITRS = 3
config.NUM_JOINTS_PER_HAND = 21


config.POSE_3D.INPUT_CHANNEL = 3
config.POSE_3D.IMAGE_SIZE = [224, 224] 
config.POSE_3D.HEAT_MAP_SCALE = 4
config.POSE_3D.HEATMAP_SIZE = [config.POSE_3D.IMAGE_SIZE[0] // config.POSE_3D.HEAT_MAP_SCALE, config.POSE_3D.IMAGE_SIZE[1] // config.POSE_3D.HEAT_MAP_SCALE]  # width * height, ex: 24 * 32
config.POSE_3D.TARGET_TYPE = 'gaussian'
config.POSE_3D.SIGMA = 2

# if config.DATASET.TRAIN_NAME == 'HOT3D':
#     config.POSE_3D.ARM = False


config.DATASET.H2O_ROOT = config.DATASET.DIR + "EgoForce/H2O"
config.DATASET.HOT3D_ROOT = config.DATASET.DIR + "EgoForce/HOT3D"
config.DATASET.ARCTIC_ROOT = config.DATASET.DIR + 'EgoForce/ARCTIC'
config.DATASET.HANDCO_ROOT = config.DATASET.DIR + 'EgoForce/HanCo'
config.DATASET.REINTERHAND_ROOT = config.DATASET.DIR + 'EgoForce/re_interhand'
config.DATASET.HO3D_ROOT = config.DATASET.DIR + 'EgoForce/HO3DV2'

if config.DATASET.TRAIN_NAME == 'HOT3D':
    config.DATASET.TRAIN_ROOT = config.DATASET.HOT3D_ROOT

elif config.DATASET.TRAIN_NAME == 'H2O':
    config.DATASET.TRAIN_ROOT = config.DATASET.H2O_ROOT

elif config.DATASET.TRAIN_NAME == 'ARCTIC':
    config.DATASET.TRAIN_ROOT = config.DATASET.ARCTIC_ROOT

elif config.DATASET.TRAIN_NAME == 'HANDCO':
    config.DATASET.TRAIN_ROOT = config.DATASET.HANDCO_ROOT

elif config.DATASET.TRAIN_NAME == 'ARCTIC_FULL':
    config.DATASET.TRAIN_ROOT = config.DATASET.ARCTIC_ROOT

elif config.DATASET.TRAIN_NAME == 'H2O_FULL':
    config.DATASET.TRAIN_ROOT = config.DATASET.H2O_ROOT

elif config.DATASET.TRAIN_NAME == 'HO3D':
    config.DATASET.TRAIN_ROOT = config.DATASET.HO3D_ROOT
else:
    config.DATASET.TRAIN_ROOT = config.DATASET.ARCTIC_ROOT

if config.DATASET.TEST_NAME == 'HOT3D':
    config.DATASET.TEST_ROOT = config.DATASET.HOT3D_ROOT
elif config.DATASET.TEST_NAME == 'H2O':
    config.DATASET.TEST_ROOT = config.DATASET.H2O_ROOT
elif config.DATASET.TEST_NAME == 'ARCTIC':
    config.DATASET.TEST_ROOT = config.DATASET.ARCTIC_ROOT
elif config.DATASET.TEST_NAME == 'FREIHAND':
    config.DATASET.TEST_ROOT = config.DATASET.HANDCO_ROOT
elif config.DATASET.TEST_NAME == 'HO3D':
    config.DATASET.TEST_ROOT = config.DATASET.HO3D_ROOT

config.GPUS = '0'
config.WORKERS = 4

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True



def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array([eval(x) if isinstance(x, str) else x
                                  for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array([eval(x) if isinstance(x, str) else x
                                 for x in v['STD']])
    if k == 'MODEL':
        if 'EXTRA' in v and 'HEATMAP_SIZE' in v['EXTRA']:
            if isinstance(v['EXTRA']['HEATMAP_SIZE'], int):
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    [v['EXTRA']['HEATMAP_SIZE'], v['EXTRA']['HEATMAP_SIZE']])
            else:
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    v['EXTRA']['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(
            config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.COCO_BBOX_FILE = os.path.join(
            config.DATA_DIR, config.TEST.COCO_BBOX_FILE)

    config.POSE_3D.PRETRAINED = os.path.join(
            config.DATA_DIR, config.POSE_3D.PRETRAINED)


def get_model_name(cfg):
    name = cfg.POSE_3D.NAME
    full_name = cfg.POSE_3D.NAME
    extra = cfg.POSE_3D.EXTRA
    if name in ['pose_resnet']:
        name = '{model}_{num_layers}'.format(
            model=name,
            num_layers=extra.NUM_LAYERS)
        deconv_suffix = ''.join(
            'd{}'.format(num_filters)
            for num_filters in extra.NUM_DECONV_FILTERS)
        full_name = '{height}x{width}_{name}_{deconv_suffix}'.format(
            height=cfg.POSE_3D.IMAGE_SIZE[1],
            width=cfg.POSE_3D.IMAGE_SIZE[0],
            name=name,
            deconv_suffix=deconv_suffix)
    else:
        raise ValueError('Unkown model: {}'.format(cfg.POSE_3D))

    return name, full_name



if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])