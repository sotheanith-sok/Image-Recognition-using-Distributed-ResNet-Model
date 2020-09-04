#Import
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import horovod.tensorflow.keras as hvd




#initialize horovod and get ranks
hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


#Hyperparameters
batch_size = 32
learning_rate = 0.001
epochs = 25

#Load data
(x_train, digit_y_train), (x_test, digit_y_test) = cifar10.load_data()

y_train = to_categorical(digit_y_train.flatten(),10)
y_test = to_categorical(digit_y_test.flatten(),10)


generator = ImageDataGenerator(rescale=1/255.)
generator.fit(x_train)




def ResnetLayers(previous_layers, filters=128, kernel_size=(3,3), padding='same', strides=1 , activation='relu'):
    #First Conv2D layer
    main_path = layers.BatchNormalization()(previous_layers)
    main_path = layers.Activation(activation)(main_path)
    main_path = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding = padding, strides = strides, kernel_initializer='he_normal')(main_path)
    
    #Second Conv2D layer
    main_path = layers.BatchNormalization()(main_path) 
    main_path = layers.Activation(activation)(main_path)
    main_path = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding = padding, kernel_initializer='he_normal')(main_path)
    

    #Check if inputs have the right shape to perform addition
    if main_path.shape[1:] == previous_layers.shape[1:]:
        shortcut_path = previous_layers
    else:
        shortcut_path = layers.Conv2D(filters=filters, kernel_size=(1,1), padding = padding, strides = strides, activation= activation, kernel_initializer='he_normal')(previous_layers)

    #Complete the second Conv2D layer
    main_path = layers.Add()([main_path, shortcut_path])
    return main_path

def Resnet(x_train):
    inputs = layers.Input(shape= x_train.shape[1:])

    #First layers
    outputs = layers.Conv2D(filters=16, kernel_size=(3,3), padding = 'same', strides = 1, kernel_initializer='he_normal')(inputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Activation('relu')(outputs)

    #Stacks 1
    outputs = ResnetLayers(outputs,  filters=16, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=16, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=16, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=16, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=16, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=16, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=16, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=16, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=16, kernel_size=(3,3), padding='same', strides=1 , activation='relu')

    #Stacks 2
    outputs = ResnetLayers(outputs,  filters=32, kernel_size=(3,3), padding='same', strides=2 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=32, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=32, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=32, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=32, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=32, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=32, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=32, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=32, kernel_size=(3,3), padding='same', strides=1 , activation='relu')


    #Stacks 3
    outputs = ResnetLayers(outputs,  filters=64, kernel_size=(3,3), padding='same', strides=2 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=64, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=64, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=64, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=64, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=64, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=64, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=64, kernel_size=(3,3), padding='same', strides=1 , activation='relu')
    outputs = ResnetLayers(outputs,  filters=64, kernel_size=(3,3), padding='same', strides=1 , activation='relu')


    outputs = layers.AveragePooling2D(pool_size=8)(outputs)
    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(10, activation="softmax", kernel_initializer='he_normal')(outputs)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

model = Resnet(x_train)

#Normal optimizers
opt = optimizers.Adam(lr=learning_rate*hvd.size())

# Horovod: add Horovod DistributedOptimizer around normal optimizer
opt = hvd.DistributedOptimizer(opt, sparse_as_dense=True, compression=hvd.Compression.fp16)

#Compile model
model.compile(
    optimizer=opt,
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

#Horovod speciall callbacks
callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, initial_lr=learning_rate*hvd.size(), verbose=1),
]

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

import time
t0=time.time()

#Train model
model.fit(generator.flow(x=x_train,y=y_train, batch_size=batch_size), steps_per_epoch=500//hvd.size(), epochs=epochs, callbacks=callbacks, verbose=verbose)

t1=time.time()
total=t1-t0

print("Total training time: ",total, " seconds")