# import the necessary packages
import os
# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "Dataset/cuhk03"
# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
#BASE_PATH = "C:\\Users\\admin\\PycharmProjects\\PersonReidResnet\\Dataset"
BASE_PATH = "Dataset/cuhk03"
# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
# VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# define the amount of data that will be used training
TRAIN_SPLIT = 0.70
# the amount of validation data will be a percentage of the
# *training* data
VAL_SPLIT = 0.30
# define the names of the classes
# CLASSES = ["person1", "person2"]

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 4
NUM_EPOCHS = 500
# define the path to the serialized output model after training
MODEL_PATH = "output/cuhk03/model/PersonReidModel.h5"

#define the path for graph plots
GRAPH_PLOT = "output/cuhk03/graph/plot"
# GRAPH_PLOT_ACC = "/home/sushmita/PycharmProjects/PersonReidResnet/output/cuhk03/graph/acc_plot"
# GRAPH_PLOT_LOSS = "/home/sushmita/PycharmProjects/PersonReidResnet/output/cuhk03/graph/loss_plot"

VISUALIZATION_PATH="output/cuhk03/rankList/final"
SIMILARIMAGE_PATH="output/cuhk03/similarImages/similar"

CHECKPOINTS_PATH="output/cuhk03/checkpoints/PersonReidModel.{epoch:02d}-{val_accuracy:.4f}.hdf5"

FONT_PATH="font/Arial Bold.ttf"

