from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, Reshape
from keras import optimizers


print('Building model...')
model = Sequential()
model.add(Dense(11, activation = 'relu', input_shape=(22,)))
model.add(Dropout(0.2))
model.add(Dense(5, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation="sigmoid"))