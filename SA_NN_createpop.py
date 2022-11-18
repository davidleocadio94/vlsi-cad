import os
import cv2
import scipy
from PIL import Image
#from tqdm import tqdm
import numpy as np
#DATADIR = "test"
CATEGORIES = ["HS", "NHS"]


IMG_SIZE = 1024

 #new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
 #plt.imshow(new_array, cmap='gray')
 #plt.show()




def create_pop_SA(n,X,y):
 pop=[]
 from sklearn.model_selection import train_test_split
 X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
 for _ in range(n):
  print("pop len is")
  print(len(pop))
  #from datetime import datetime
  #start=datetime.now()
  #times=[]
  max_neurons=200
  max_conv_kernel=3
  hyperparameters={}
  hyperparameters2={}
  T0=0.15
  Tf=.035
  T=T0
  Ts=[]
  accuracies=[]
  max_iter=np.exp(T0/Tf).astype(int)
  neurons_dense_list=[]
  conv_output_channels_list=[]
  conv_kernel_sizes_list=[]
  steps=[]
  epochs=[]
  for i in range(max_iter):
          Ts.append(T)
          iteration=i
          if i==0:
              n_epochs=np.random.randint(1,15,1)[0]
              n_conv_layers=np.random.randint(1,10,1)[0]
              conv_output_channels=np.random.randint(1,10,n_conv_layers)
              conv_kernel_sizes=np.random.randint(1,10,n_conv_layers)
              n_dense_layers=np.random.randint(1,10,1)[0]
              neurons_dense=np.random.randint(1,400,n_dense_layers)
              hyperparameters['n_epochs']=n_epochs
              hyperparameters['n_dense_layers']=n_dense_layers
              hyperparameters['n_conv_layers']=n_conv_layers
              hyperparameters['neurons_dense']=neurons_dense
              hyperparameters['conv_output_channels']=conv_output_channels
              hyperparameters['conv_kernel_sizes']=conv_kernel_sizes
              hyperparameters['val_accuracy']=0
          elif i>0:
              n_epochs=hyperparameters['n_epochs']
              n_dense_layers=hyperparameters['n_dense_layers']
              n_conv_layers=hyperparameters['n_conv_layers']
              neurons_dense=hyperparameters['neurons_dense']
              conv_output_channels=hyperparameters['conv_output_channels']
              conv_kernel_sizes=hyperparameters['conv_kernel_sizes']
              n_dense_layers2=(np.random.randn(1)[0]*5+n_dense_layers).astype(int)
              n_conv_layers2=(np.random.randn(1)[0]*5+n_conv_layers).astype(int)
              n_epochs=(np.random.randn(1)[0]*1*n_epochs+n_epochs).astype(int)
              if n_epochs<1:
                  n_epochs=2
              if n_epochs>7:
                  n_epochs=5
              if n_dense_layers2<1:
                  n_dense_layers2=1
              if n_dense_layers2>8:
                  n_dense_layers2=5
              if n_conv_layers2<1:
                  n_conv_layers2=1
              if n_conv_layers2>9:
                  n_conv_layers2=5            
              neurons_dense=(np.random.randn(n_dense_layers)*5+neurons_dense).astype(int)
              neurons_dense[neurons_dense<1]=10
              neurons_dense[neurons_dense>200]=50
              neurons_dense2=np.random.randint(20,max_neurons,n_dense_layers2)
              neurons_dense2[:np.min([n_dense_layers,n_dense_layers2])]=neurons_dense[:np.min([n_dense_layers,n_dense_layers2])]
              neurons_dense=neurons_dense2
              n_dense_layers=n_dense_layers2
              neurons_dense_list.append(hyperparameters['neurons_dense'])
              
  
              conv_output_channels=(np.random.randn(n_conv_layers)*1+conv_output_channels).astype(int)
              conv_output_channels[conv_output_channels<1]=1
              conv_output_channels[conv_output_channels>7]=5
              conv_output_channels2=np.random.randint(1,5,n_conv_layers2)
              conv_output_channels2[:np.min([n_conv_layers,n_conv_layers2])]=conv_output_channels[:np.min([n_conv_layers,n_conv_layers2])]
              conv_output_channels=conv_output_channels2
              conv_output_channels_list.append(hyperparameters['conv_output_channels'])            
              
                
              conv_kernel_sizes=(np.random.randn(n_conv_layers)*1+conv_kernel_sizes).astype(int)
              conv_kernel_sizes[conv_kernel_sizes<1]=1
              conv_kernel_sizes[conv_kernel_sizes>7]=5
              conv_kernel_sizes2=np.random.randint(1,5,n_conv_layers2)
              conv_kernel_sizes2[:np.min([n_conv_layers,n_conv_layers2])]=conv_kernel_sizes[:np.min([n_conv_layers,n_conv_layers2])]
              conv_kernel_sizes=conv_kernel_sizes2
              n_conv_layers=n_conv_layers2
              conv_kernel_sizes_list.append(hyperparameters['conv_kernel_sizes'])           
              epochs.append(n_epochs)
              
          import tensorflow as tf
          from tensorflow.keras.datasets import cifar10
          from tensorflow.keras.preprocessing.image import ImageDataGenerator
          from tensorflow.keras.models import Sequential
          from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
          from tensorflow.keras.layers import Conv2D, MaxPooling2D
  
          model = Sequential()
  
          #model.add(Conv2D(4, (3, 3),strides=(2,2), input_shape=X.shape[1:]))
          #model.add(Activation('relu'))
          #model.add(MaxPooling2D(pool_size=(2, 2)))
  
          #model.add(Conv2D(4, (3, 3)))
          #model.add(Activation('relu'))
          #model.add(MaxPooling2D(pool_size=(2, 2)))
          #print(conv_output_channels)
          #print(conv_kernel_sizes)
          for j in range(n_conv_layers):
              print(conv_output_channels[j])
              print(conv_kernel_sizes[j])
              model.add(Conv2D(conv_output_channels[j],(conv_kernel_sizes[j],conv_kernel_sizes[j]),padding="same"))
              model.add(Activation('relu'))
              model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
          
          
  
          model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
          
          for j in neurons_dense:
              model.add(Dense(j))
              model.add(Activation('relu'))
              
          model.add(Dense(1))
          model.add(Activation('sigmoid'))
  
  
          model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
          print("N NEURONS",neurons_dense.shape)
          print(neurons_dense)
          history=model.fit(X_train, y_train, batch_size=2, epochs=n_epochs, validation_data=(X_test,y_test))
          
          val_accuracy=history.history['val_accuracy'][-1]
          
          p=np.random.uniform(0,1,1)
          if val_accuracy-hyperparameters['val_accuracy']>0:
              hyperparameters['n_epochs']=n_epochs
              hyperparameters['n_dense_layers']=n_dense_layers
              hyperparameters['n_conv_layers']=n_conv_layers
              hyperparameters['neurons_dense']=neurons_dense
              hyperparameters['val_accuracy']=val_accuracy
              hyperparameters['conv_output_channels']=conv_output_channels
              hyperparameters['conv_kernel_sizes']=conv_kernel_sizes
              steps.append("1")
          elif np.exp(-(hyperparameters['val_accuracy']-val_accuracy)/T)>=p:
              hyperparameters['n_epochs']=n_epochs
              hyperparameters['n_dense_layers']=n_dense_layers
              hyperparameters['n_conv_layers']=n_conv_layers
              hyperparameters['neurons_dense']=neurons_dense
              hyperparameters['val_accuracy']=val_accuracy
              hyperparameters['conv_output_channels']=conv_output_channels
              hyperparameters['conv_kernel_sizes']=conv_kernel_sizes
              steps.append("2")
          else:
              print("passed")
              steps.append("3")
          T=T0/(1+np.log(1+iteration*1))
          accuracies.append(hyperparameters['val_accuracy'])
          #times.append((datetime.now()-start).seconds)
          if (np.max(accuracies)>0.7):
             break
  
  pop.append(hyperparameters)
 return pop
