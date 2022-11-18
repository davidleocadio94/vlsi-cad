import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import random
#import multiprocessing
#pool=multiprocessing.Pool()

from load_data import * 
from gen_pop import * 
from train_and_score import * 
from select_breed_mutate import * 
from SA_NN_createpop import *

from datetime import datetime
start=datetime.now()



X,y=create_data("test",[])
#np.save('X',X)
#np.save('y',y)
#pop = create_pop(10)
pop = create_pop_SA(6,X,y)
avgscoreslist=[]
times=[]
for _ in range(1):
    scores=train_score(pop,X,y)
    avgscoreslist.append(np.max(scores))
    print(avgscoreslist)
    if np.max(avgscoreslist)>0.97:
      break
    times.append((datetime.now()-start).seconds)
    print("doing for loop")
    pop=select_breed_mutate(pop,scores,.4)
print(scores)
print(avgscoreslist)
print(times)
