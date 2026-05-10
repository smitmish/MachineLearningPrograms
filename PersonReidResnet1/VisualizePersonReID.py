# ============================== Visualize Part=====================================
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import config

img_height, img_width = (224, 224)
rank = 10
num_of_images_to_show_from_each_folder = 1
query_image = 1

test_data_dir = config.TEST_PATH
train_data_dir = config.TRAIN_PATH

datagen = ImageDataGenerator()
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
datagen.mean = mean

test_generator = datagen.flow_from_directory(test_data_dir, target_size=(img_height, img_width),
                                             batch_size=1, class_mode="categorical", shuffle=False)
train_generator = datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width),
                                              batch_size=config.BS, class_mode="categorical", shuffle=False)
# print(test_generator.classes) #classes and labels are same [0 0 0 1 1 1 1 1]
# print(test_generator.class_indices) #{'person1': 0, 'person2': 1}

test_generator.reset()
class_names = test_generator.class_indices
# print(type(class_names))
# print(class_names)
train_class_names = train_generator.class_indices


# print(train_class_names)


# This function is for listing the predicted class names in descending order of probabilities
# Argument: Predicted Probabilities of one Image
# Return: list of class names in decreasing order of Probabilities
# def classProbMap(predProbs):
#     classNameProbMapList = []
#     for k in range(len(predProbs)):
#         classNameProbMapList.append([list(test_generator.class_indices.keys())[k], predProbs[k]])
#     predictedClassLabels = list(np.array(sorted(classNameProbMapList, key=lambda l: l[1], reverse=True))[:, 0])
#     return predictedClassLabels


def predict_img(num_images):
    y_predicted = []
    y_act = []
    filelist = []
    model = tf.keras.models.load_model(config.MODEL_PATH)
    print(model)
    for i in range(0, num_images):
        X_test, Y_test = test_generator.next()
        # print(test_generator.filenames[i])
        print("X_test",type(X_test))  #X_test <class 'numpy.ndarray'>
        # print("Y_test",type(Y_test))
        # print(Y_test)
        # print(test_generator.class_indices)
        # print(test_generator.classes[i])
        features = model.predict(X_test)
        # print("features",features)
        flattened_features = features.flatten()
        # print("flattened_features", len(flattened_features))
        # print(test_generator.filepaths[i])
        filelist.append(test_generator.filepaths[i])
        # y_predicted.append(classProbMap(flattened_features))
        top_values_index = sorted(range(len(flattened_features)), key=lambda i: flattened_features[i])[-rank:]
        # print("top_values_index :::", top_values_index)
        # top_values = [flattened_features[i] for i in np.argsort(flattened_features)[-5:]]
        # print("top_values :: ",top_values)
        y_pred5Class = [list(train_generator.class_indices.keys())[i] for i in top_values_index]
        # print("top 5 pred :: ",y_pred5Class)
        reversed_list = y_pred5Class[::-1]
        # print("top 5 pred reverse :: ", reversed_list)
        y_predicted.append(reversed_list)
        y_act.append(Y_test)
        # print(Y_test)
        # print("actual :: ", [list(test_generator.class_indices.keys())[i.argmax()] for i in Y_test])
    # print(y_predicted)
    actual_class = [list(test_generator.class_indices.keys())[i.argmax()] for i in y_act]
    # print(actual_class)
    return filelist, y_predicted, actual_class


def load_images_from_folder(folderlist, qimage, actualClass):
    imageLst = []
    imageLst.append(qimage)
    color = "green"
    for folder in range(len(folderlist)):
        # path = "/home/sushmita/PycharmProjects/ResnetPerson/fineTuneResnet/finalImageData/training/" + str(
        # folderlist[folder]) + "/"
        path = train_data_dir + "\\" + str(folderlist[folder]) + "\\"
        # print(path)
        count = 0
        # print(actualClass)
        # print(str(folderlist[folder]))
        if actualClass != str(folderlist[folder]):
            color = "red"
        else:
            color = "green"
        for filename in os.listdir(path):
            # img = cv2.imread(os.path.join(path,filename))
            if count < num_of_images_to_show_from_each_folder:
                img = Image.open(os.path.join(path, filename))
                # print(os.path.join(path, filename))
                if img is not None:
                    # dstimg = cv2.resize(img,(250,250))
                    # img = img.resize((250, 250))
                    img = ImageOps.expand(img, border=5, fill=color)
                    imageLst.append(img)
                    count = count + 1
    return imageLst


def merge_images_horizontally(imgs, count):
    '''
    This function merges images horizontally.
    '''
    # create two lists - one for heights and one for widths
    widths, heights = zip(*(i.size for i in imgs))
    width_of_new_image = sum(widths)
    height_of_new_image = min(heights)  # take minimum height
    # create new image
    new_im = Image.new('RGB', (width_of_new_image, height_of_new_image))
    new_pos = 0
    for im in imgs:
        new_im.paste(im, (new_pos, 0))
        new_pos += im.size[0]  # position for the next image
    new_im.save(config.VISUALIZATION_PATH + count + '.jpg')


if __name__ == '__main__':
    # query_image = len(test_generator.filenames)
    print("Total file are:",query_image)
    filelist, y_predicted, y_act = predict_img(query_image)
    # print(filelist,y_predicted,y_act)
    images = []
    for i in range(0, query_image):
        # print(filelist[i])
        # print(y_predicted[i][0])
        # print(y_act[i])
        imageRow = []
        img = Image.open(filelist[i])
        # img = img.resize((250, 250))
        # print(type(y_predicted[i]))
        # print(y_predicted[i][0:3])
        # top3predVal = y_predicted[i][0:3]
        imageRow = load_images_from_folder(y_predicted[i], img, y_act[i])
        # print(imageRow)
        images.append(imageRow)
    for i in range(len(images)):
        merge_images_horizontally(images[i], str(i + 1))
