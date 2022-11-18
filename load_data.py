import numpy as np
#import matplotlib.pyplot as plt
import os
import cv2
import scipy
from PIL import Image

import random
#from tqdm import tqdm

#DATADIR = "test"

CATEGORIES = ["HS", "NHS"]


IMG_SIZE = 1024

#new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#plt.imshow(new_array, cmap='gray')
#plt.show()



#training_data = []
def create_data(DATADIR,data):
    for category in CATEGORIES:  # do dogs and cats
        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
        k=0
        for img in os.listdir(path):  # iterate over each image per dogs and cats
                #img_array = Image.open(os.path.join(path,img)).convert('LA')  # convert to array
                #new_array = img_array.resize((IMG_SIZE,IMG_SIZE))  # resize to normalize data size
                #new_array=np.asarray(new_array)[:,:,0]
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                data.append([new_array, class_num])  # add this to our training_data
                k+=1
                if k>500:
                  break
    #return data
        





    random.shuffle(data)

    for sample in data[:10]:
        print(sample[1])




    X = []
    y = []

    for features,label in data:
        X.append(features)
        y.append(label)




    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype(np.float)
    y = np.array(y)
    print(X[0].shape)
    print(y.shape)
    return X,y

