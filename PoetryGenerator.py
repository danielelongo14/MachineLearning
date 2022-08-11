import keras.optimizers
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.layers import LSTM, Dense, Activation
from keras.models import Sequential
import random


optimizers.RMSprop


filepath = tf.keras.utils.get_file("shakespeare","https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")

text = open(filepath,"rb").read().decode(encoding="utf-8").lower()

text =  text[0:800000]

characters =  sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i,c) for i,c in enumerate(characters))

SEQ_LENGHT = 40
STEP_SIZE = 3




sentences = []
next_characters = []

for i in range(0, len(text)- SEQ_LENGHT, STEP_SIZE):
    sentences.append(text[i: i+SEQ_LENGHT])
    next_characters.append(text[i+SEQ_LENGHT])

x = np.zeros((len(sentences), SEQ_LENGHT, len(characters)), dtype=np.bool)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

for i, sentence in enumerate(sentences):                     #for each of the sentences we enumerate the sentence and setting the position for each space equals to 1
    for t, character in enumerate(sentence):
        x[i,t,char_to_index[character]] = 1
    y[i,char_to_index[next_characters[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGHT, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",optimizer= keras.optimizers.RMSprop(lr=0.01))

model.fit(x,y,batch_size=256,epochs=4)

model.save("textgenerator.model")


model = keras.models.load_model("textgenerator.model")

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds,1)
    return np.argmax(probas)

def generate_text(lenght, temperature):
    start_index = random.randint(0,len(text) - SEQ_LENGHT - 1)
    generated = ""
    sentence = text[start_index : start_index + SEQ_LENGHT]
    generated += sentence
    for i in range(lenght):
        x =  np.zeros((1,SEQ_LENGHT,len(characters)))
        for t, character in enumerate(sentence):
            x[0,t,char_to_index[character]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character

    return generated


print("0.2")
print(generate_text(300,0.2))

print("0.5")
print(generate_text(300,0.5))

print("0.8")
print(generate_text(300,0.8))

print("1")
print(generate_text(300,1.0))