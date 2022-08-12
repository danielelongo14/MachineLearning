import cv2 as cv
import keras.datasets.cifar10
import numpy as np
import matplotlib.pyplot as plt
from keras import  datasets, layers, models


#Preparing Data
(training_images,training_labels), (testing_imgaes,testing_labels) = keras.datasets.cifar10.load_data()
training_images, testing_imgaes = training_images/255, testing_imgaes/255


#Create a list for labels
class_name = ["Plane","Car","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"]

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_name[training_labels[i][0]])

plt.show()

#create the model

model = keras.models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu", input_shape=(32,32,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(training_images,training_labels, epochs=10, validation_data= (testing_imgaes,testing_labels))

loss, accuracy = model.evaluate(testing_imgaes,testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save("image_classifier.model")



model = models.load_model("image_classifier.model")

img = cv.imread("deer.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f"Prediction is {class_name[index]}")

plt.show()