
import tensorflow as tf
print(tf.__version__)
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(7)

inputs = []
outputs = []

with open("2025_Veriseti_YZ.csv") as f:
    f.readline()
    for i in f:
        i = i[:-1]
        i = i.split(",")
        inp = [int(data) for data in i[1:-1]]
        out = [int(i[-1][0])-1]
        inputs.append(inp)
        outputs.append(out)

inputs = np.array(inputs)
outputs = np.array(outputs)

model = Sequential(
    [
        Dense(600, input_shape=(10,) ,activation='relu'),
        Dense(450, activation='relu'),
        Dense(100, activation='sigmoid'),
        Dense(1, activation='sigmoid')
    ]
)

traininp,testinp,trainout,testout = train_test_split(inputs,outputs, test_size=0.2, random_state=42)

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(traininp, trainout, epochs=500, batch_size=3200)

train_basarim = model.evaluate(traininp, trainout)
print("\nEðitim Baþarýmý: %s : %.2f%%" % (model.metrics_names[1], train_basarim[1] * 100))


test_basarim = model.evaluate(testinp, testout)
print("\nTest Baþarýmý: %s : %.2f%%" % (model.metrics_names[1], test_basarim[1] * 100))
