import tensorflow as tf
#from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np


def train_score(population,X,y):
        #X=np.load('X.npy')
        #y=np.load('y.npy')
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    scores=[]
        #member=population
    for member in population:
        
        n_epochs=member['n_epochs']
        n_dense_layers=member['n_dense_layers']
        n_conv_layers=member['n_conv_layers']
        neurons_dense=member['neurons_dense']
        conv_output_channels=member['conv_output_channels']
        conv_kernel_sizes=member['conv_kernel_sizes']

        model = Sequential()

        for j in range(n_conv_layers):
            #print(conv_output_channels[j])
            #print(conv_kernel_sizes[j])
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
        print("N CONV LAYERS",n_conv_layers)
        print("CONV OUTPUT CHANNELS",conv_output_channels)
        print("conv_kernel_sizes",conv_kernel_sizes)
        history=model.fit(X_train, y_train, batch_size=80, epochs=n_epochs, validation_data=(X_test,y_test))
        val_accuracy=history.history['val_accuracy'][-1]        
        scores.append(val_accuracy)


        #scores=val_accuracy
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn import metrics
        from sklearn.metrics import plot_roc_curve
        y_pred = model.predict(X_test)
        y_pred = y_pred>.5
        print(classification_report(y_test, y_pred))

        from sklearn.metrics import roc_curve
        y_pred_keras = model.predict(X_test).ravel()
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
        from sklearn.metrics import auc
        auc_keras = auc(fpr_keras, tpr_keras)


        print("fpr_keras")
        print(fpr_keras)
        print("tpr_keras")
        print(tpr_keras)
        print("auc")
        print(auc_keras)
    return scores

