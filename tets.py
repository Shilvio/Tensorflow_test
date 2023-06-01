import tensorflow as tf
# import matplotlib.pyplot as plt

# loads training and test data in touples
(train_images, train_labels), (test_images,
                               test_labels) = tf.keras.datasets.mnist.load_data()

# images have pixel values that goes from 0-255, this converts the value into floats between 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

# data visualization
print(train_images.shape)
print(test_images.shape)
print(train_labels)

# display first image
# plt.imshow(train_images[0], cmap='gray')
# plt.show()

# define neural network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model compilation
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model training
model.fit(train_images, train_labels, epochs=10)

# accuracy checking
val_loss, val_acc = model.evaluate(test_images, test_labels)
print('Accuracy results : ', val_acc)

# save model
model.save("mnist_test")

# load model from file-system
test_model = tf.keras.models.load_model('mnist_test')

# performance checking
saved_loss, saved_acc = model.evaluate(test_images, test_labels)
print('Accuracy comparison saved : ', saved_acc, ' old : ', saved_acc)
