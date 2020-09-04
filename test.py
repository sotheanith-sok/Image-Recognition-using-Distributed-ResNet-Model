import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x,y),_ = mnist.load_data()

x = x[...,tf.newaxis]

x=x[0:10]
y = y[0:10]

for i in range(10):
    x[i,:]=i

print(x[1])
generator = ImageDataGenerator()
generator.fit(x)

dataset = tf.data.Dataset.from_tensor_slices([[1,2],[3,4],[5,6],[7,8],[9,10]])



for i in dataset:
    print(i)

# dataset=tf.data.Dataset.from_generator(generator.flow,args=[x,y,10],output_types=(tf.float64,tf.int8))



# dataset.batch(1)

# for i in dataset:
#     print(np.shape(i))

# dataset=tf.data.Dataset.range(120)
# dataset = dataset.batch(11)
# a = list(dataset.as_numpy_iterator())
# print(a)