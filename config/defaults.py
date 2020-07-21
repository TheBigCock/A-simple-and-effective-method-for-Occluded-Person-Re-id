from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50_ibn_a'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone   "../modeling/backbones/resnet50_ibn_a.pth"
_C.MODEL.PRETRAIN_PATH =  "../modeling/backbones/resnet50_ibn_a.pth"
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
# If train with STN, options: 'yes' or 'no'
_C.MODEL.BREACH = 'no'
# The loss type of metric loss
# options:['triplet'](without center loss) or ['center','triplet_center'](with center loss)
#  度量损失
_C.MODEL.METRIC_LOSS_TYPE = 'ranked_loss'

# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training [384, 128]
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST =[256, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
_C.INPUT.CUT_PROB = 0.5
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')  # dukemtmc
# Root directory where data_reader should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data/datasets')

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax_rank'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 120
# Base learning rate
_C.SOLVER.BASE_LR = 0.00035
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1

# Margin of ranked list loss
_C.SOLVER.MARGIN_RANK = 1.3  ### R = ALPHA - MARGIN_RANK
_C.SOLVER.ALPHA = 2.0
_C.SOLVER.TVAL = 1.0
_C.SOLVER.WEIGHT = 0.3       ### loss = softmax + w*ranked_loss  0.3

# ms loss
_C.SOLVER.MULTI_SIMILARITY_LOSS = CN()
_C.SOLVER.MULTI_SIMILARITY_LOSS.SCALE_POS = 3.0
_C.SOLVER.MULTI_SIMILARITY_LOSS.SCALE_NEG = 60.0
_C.SOLVER.MULTI_SIMILARITY_LOSS.HARD_MINING = True

# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Margin of cluster ;pss
_C.SOLVER.CLUSTER_MARGIN = 0.3

# Settings of range loss
_C.SOLVER.RANGE_K = 2
_C.SOLVER.RANGE_MARGIN = 0.3
_C.SOLVER.RANGE_ALPHA = 0
_C.SOLVER.RANGE_BETA = 1
_C.SOLVER.RANGE_LOSS_WEIGHT = 1

_C.SOLVER.ATTENTION = 0.2   # 0.2

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = [40, 70]

# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
# iterations of warm up
_C.SOLVER.WARMUP_ITERS = 10
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 20
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 20
# epoch number of validation验证
_C.SOLVER.EVAL_PERIOD = 120

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'yes','no'
_C.TEST.RE_RANKING = 'no'
# Path to trained model
_C.TEST.WEIGHT = "../logfile/market1501/ranked_loss23/resnet50_ibn_a_model_40.pth"

# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model  dukemtmc market1501
_C.OUTPUT_DIR = "../logfile/market1501/ranked_loss23/test"
