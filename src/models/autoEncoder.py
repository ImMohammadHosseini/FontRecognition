"""
"""


import keras
from keras import layers

def AutoEncoder (input_shape):
    model = keras.Sequential([
        layers.Input(shape = input_shape),
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(128, kernel_size=(2, 2), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), name='encoded'),

        layers.Conv2DTranspose(128, (2, 2), activation = 'relu', padding='same'),
        layers.BatchNormalization(),
        layers.UpSampling2D(size=(2, 2)),

        layers.Conv2DTranspose(128, (3, 3), activation = 'relu', padding='same'),
        layers.UpSampling2D(size=(2, 2)),
        layers.BatchNormalization(),

        layers.Conv2DTranspose(128, (3, 3), activation = 'relu', padding='same'),
        layers.UpSampling2D(size=(2, 2)),
        layers.BatchNormalization(),

        layers.Conv2DTranspose(64, (5, 5), activation = 'relu', padding='same'),
        layers.UpSampling2D(size=(2, 2)),
        layers.BatchNormalization(),

        layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same'
                               , name='decoded')
        ]
    )
    
    model.compile(optimizer='adam', loss='mse')
    
    return model
        
