# set the matplotlib backend so figures can be saved in the background
import matplotlib
import ml_metrics as ml_metrics

matplotlib.use("Agg")
# import the necessary packages
import config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse
import tensorflow as tf


def extractDigits(lst):
	res = []
	for el in lst:
		nes = []
		nes.append(el)
		res.append(nes)
	return (res)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# determine the total number of image paths in training, validation,
# and testing directories
totalTest = len(list(paths.list_images(config.TEST_PATH)))

# initialize the validation/testing data augmentation object (which
# we'll be adding mean subtraction to)
valAug = ImageDataGenerator()
# define the ImageNet mean subtraction (in RGB order) and set the
# the mean subtraction value for each of the data augmentation
# objects
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
valAug.mean = mean

# initialize the testing generator
testGen = valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=config.BS)

# model = tf.keras.models.load_model(config.MODEL_PATH)
model = tf.keras.models.load_model(config.MODEL_PATH)
# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] model predicting...")
testGen.reset()
predIdxs1 = model.predict(testGen, steps=(totalTest // config.BS) + 1)
# print(predIdxs1)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs1, axis=1)
# print(predIdxs)
# print(testGen.labels)
# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys(),zero_division=1))

# Finding Rank 1 Accuraccy
score=0
for i in range(len(predIdxs1)):
  if np.argmax(predIdxs1[i]) == testGen.labels[i]:
    score+=1
# print(score)
# print(len(testGen.labels))
rank1 = score/float(len(testGen.labels)) *100
print("rank1 accuracy",round(rank1,2))

# Finding Rank 5 Accuraccy
score=0
predicted5List = []
for i in range(len(predIdxs1)):
	predicted5arr = np.argsort(predIdxs1[i])[::-1][:5]
	predicted5List.append(list(predicted5arr))
	if testGen.labels[i] in np.argsort(predIdxs1[i])[::-1][:5]:
		score+=1
rank5 = score/float(len(testGen.labels)) *100
# print("rank5 accuracy",round(rank5,2))
print("rank5 accuracy",round(rank5,2))

# Finding Rank 10 Accuraccy
score=0
predicted10List = []
for i in range(len(predIdxs1)):
	predicted10arr = np.argsort(predIdxs1[i])[::-1][:10]
	predicted10List.append(list(predicted10arr))
	if testGen.labels[i] in np.argsort(predIdxs1[i])[::-1][:10]:
		score+=1
rank10 = score/float(len(testGen.labels))*100
print("rank10 accuracy",round(rank10,2))
print("rank10 accuracy",round(rank10,2))


actualList = testGen.labels.tolist()
predictedList = predIdxs.tolist()
actualList = extractDigits(actualList)
predictedList = extractDigits(predictedList)
map = ml_metrics.mapk(actualList, predictedList, 1)
map = map*100
print("mAP for top 1",round(map,2))

map5 = ml_metrics.mapk(actualList, predicted5List, 5)
map5 = map5*100
print("mAP for top 5",round(map5,2))

map10 = ml_metrics.mapk(actualList, predicted10List, 10)
map10 = map10*100
print("mAP for top 10",round(map10,2))



