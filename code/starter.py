import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, GlobalMaxPooling2D, Dense


def load_and_format(in_path):
    out_df = pd.read_json(in_path)
    out_images = out_df.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)
    out_images = np.stack(out_images).squeeze()
    return out_df, out_images

# load datasets
train_df, train_images = load_and_format('../input/train.json')
print('training', train_df.shape, 'loaded', train_images.shape)
test_df, test_images = load_and_format('../input/test.json')
print('testing', test_df.shape, 'loaded', test_images.shape)

# split the train dataset 50%-50% (train-test)
X_train, X_test, y_train, y_test = train_test_split(train_images,
                                                   to_categorical(train_df['is_iceberg']),
                                                    random_state = 2017,
                                                    test_size = 0.5
                                                   )
print('Train', X_train.shape, y_train.shape)
print('Validation', X_test.shape, y_test.shape)


# simple CNN
simple_cnn = Sequential()
simple_cnn.add(BatchNormalization(input_shape = (75, 75, 2)))
for i in range(4):
    simple_cnn.add(Conv2D(8*2**i, kernel_size = (3,3)))
    simple_cnn.add(MaxPooling2D((2,2)))
simple_cnn.add(GlobalMaxPooling2D())
simple_cnn.add(Dropout(0.5))
simple_cnn.add(Dense(8))
simple_cnn.add(Dense(2, activation = 'softmax'))
simple_cnn.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
simple_cnn.summary()

# training
simple_cnn.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 17, shuffle = True)

# make predictions
test_predictions = simple_cnn.predict(test_images)

# save the predictions csv file
pred_df = test_df[['id']].copy()
pred_df['is_iceberg'] = test_predictions[:,1]
pred_df.to_csv('predictions.csv', index = False)