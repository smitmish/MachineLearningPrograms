import os, random
import shutil

m = 3
n = 3

src_dir = "/home/sushmita/PycharmProjects/PersonReidResnet/Dataset/cuhk03/training/"
dst_dir = "/home/sushmita/PycharmProjects/PersonReidResnet/Dataset/cuhk03/testing/"

for folder in os.listdir(src_dir):
    print(folder)
    file_list = os.listdir(src_dir + '/' + folder)
    # print(file_list)
    a = random.choice(file_list)
    print(a)
    b = random.choice(file_list)
    print(b)
    if not os.path.exists(dst_dir + folder):
        os.makedirs(dst_dir + folder)
    shutil.copy(src_dir + folder + "/" + a, dst_dir + folder + "/" + a)
    shutil.copy(src_dir + folder + "/" + a, dst_dir + folder + "/" + b)
