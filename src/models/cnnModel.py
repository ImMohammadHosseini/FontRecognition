
"""

"""
import keras
from keras import layers

def CNNModel (autoEncoderModel):
    cnn_model = keras.Sequential() 
    for layer in autoEncoderModel.layers[:13]:
        layer.trainable=False
        cnn_model.add(layer)

    #Cs Layers
    cnn_model.add(layers.Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same'))
    cnn_model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    cnn_model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    cnn_model.add(layers.Flatten())
    
    cnn_model.add(layers.Dense(2048, activation='relu'))
    cnn_model.add(layers.Dropout(0.5))
    
    cnn_model.add(layers.Dense(1024,activation='relu'))
    cnn_model.add(layers.Dropout(0.5))
        
    cnn_model.add(layers.Dense(512,activation='relu'))
    cnn_model.add(layers.Dropout(0.5))
    
    cnn_model.add(layers.Dense(48, activation='softmax'))
    
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    
    cnn_model.compile(optimizer=sgd, loss='mean_squared_error', 
                      metrics=['accuracy'])
    
    return cnn_model