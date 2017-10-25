import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from models import Models


class Iceberg:
    def __init__(self, height=75, width=75, batch_size=32, max_epochs=100, base_model='simple', num_classes=2):
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.base_model = base_model
        self.num_classes = num_classes
        self.load_data()

    def load_data(self):
        def load_and_format(in_path):
            out_df = pd.read_json(in_path)
            out_images = out_df.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)
            out_images = np.stack(out_images).squeeze()
            return out_df, out_images

        # load datasets
        self.train_df, self.train_images = load_and_format('../input/train.json')
        print('training', self.train_df.shape, 'loaded', self.train_images.shape)
        self.test_df, self.test_images = load_and_format('../input/test.json')
        print('testing', self.test_df.shape, 'loaded', self.test_images.shape)



    def train(self):  # split the train dataset 50%-50% (train-validation)
        X_train, X_val, y_train, y_val = train_test_split(self.train_images,
                                                            to_categorical(self.train_df['is_iceberg']),
                                                            random_state=2017,
                                                            test_size=0.2
                                                            )
        print('Train', X_train.shape, y_train.shape)
        print('Validation', X_val.shape, y_val.shape)


        callbacks = [ModelCheckpoint(filepath='../weights/best_weights_{}.hdf5'.format(self.base_model),
                                             save_best_only=True,
                                             save_weights_only=True),
                             ReduceLROnPlateau(factor=0.5,
                                               patience=2,
                                               verbose=1,
                                               epsilon=1e-4),
                             EarlyStopping(min_delta=1e-4,
                                           patience=4,
                                           verbose=1)]

        models = Models(input_shape=(self.height, self.width, X_train.shape[-1]), classes=self.num_classes)
        if self.base_model == 'vgg16':
            models.vgg16()
        elif self.base_model == 'vgg19':
            models.vgg19()
        elif self.base_model == 'resnet50':
            models.resnet50()
        elif self.base_model == 'inceptionV3':
            models.inceptionV3()
        elif self.base_model == 'simple':
            models.simple()
        else:
            print('Uknown base model')
            raise SystemExit

        # models.compile(optimizer=RMSprop(lr=1e-5))

        models.compile(optimizer=Adam())

        self.model = models.get_model()

        nTrain = len(X_train)
        nVal = len(X_val)
        print("# training images: ", nTrain)
        print("# validation images: ", nVal)

        train_datagen = ImageDataGenerator(
            zca_whitening=True,
            horizontal_flip=True,
            vertical_flip=True
        )

        def train_generator():
            while True:
                for start in range(0, nTrain, self.batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + self.batch_size, nTrain)

                    for i in range(start, end):
                        x_batch.append(train_datagen.random_transform(X_train[i]))
                        y_batch.append(y_train[i])

                    # for id in ids_train_batch.values:
                    #     # j = np.random.randint(self.nAug)
                    #     img = cv2.imread(INPUT_PATH + 'train_hq/{}.jpg'.format(id))
                    #     img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
                    #     # img = transformations2(img, j)
                    #     mask = np.array(Image.open(INPUT_PATH + 'train_masks_fixed/{}_mask.gif'.format(id)), dtype=np.uint8)
                    #     mask = cv2.resize(mask, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
                    #     # mask = transformations2(mask, j)
                    #     img = randomHueSaturationValue(img,
                    #                                    hue_shift_limit=(-50, 50),
                    #                                    sat_shift_limit=(-5, 5),
                    #                                    val_shift_limit=(-15, 15))
                    #     img, mask = randomShiftScaleRotate(img, mask,
                    #                                        shift_limit=(-0.0625, 0.0625),
                    #                                        scale_limit=(-0.1, 0.1),
                    #                                        rotate_limit=(-0, 0))
                    #     img, mask = randomHorizontalFlip(img, mask)
                    #     if self.factor != 1:
                    #         img = cv2.resize(img, (self.input_dim//self.factor, self.input_dim//self.factor), interpolation=cv2.INTER_LINEAR)
                    #     # draw(img, mask)
                    #
                    #     if self.direct_result:
                    #         mask = np.expand_dims(mask, axis=2)
                    #         x_batch.append(img)
                    #         y_batch.append(mask)
                    #     else:
                    #         target = np.zeros((mask.shape[0], mask.shape[1], self.nb_classes))
                    #         for k in range(self.nb_classes):
                    #             target[:,:,k] = (mask == k)
                    #         x_batch.append(img)
                    #         y_batch.append(target)
                    #
                    # x_batch = np.array(x_batch, np.float32) / 255.0
                    # y_batch = np.array(y_batch, np.float32)
                    x_batch = np.array(x_batch, np.float32)
                    y_batch = np.array(y_batch, np.float32)
                    yield x_batch, y_batch


        # train_gen = train_datagen.flow(
        #     X_train, y_train, shuffle=True, batch_size=self.batch_size
        # )
        # train_gen = train_datagen.flow_from_directory(
        #     self.train_images,
        #     target_size=(self.height, self.width),
        #     batch_size=self.batch_size,
        #     class_mode='binary'
        # )
        # self.train_gen = Iterator(train_bson_file, train_images_df, train_offsets_df,
        #                          self.num_classes, train_datagen, lock,
        #                          batch_size=self.batch_size, shuffle=True)

        val_datagen = ImageDataGenerator()
        def val_generator():
            while True:
                for start in range(0, nVal, self.batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + self.batch_size, nVal)

                    for i in range(start, end):
                        x_batch.append(val_datagen.random_transform(X_val[i]))
                        y_batch.append(y_val[i])
                    x_batch = np.array(x_batch, np.float32)
                    y_batch = np.array(y_batch, np.float32)
                    yield x_batch, y_batch
        # val_gen = val_datagen.flow(
        #     X_test, y_test, batch_size=self.batch_size, save_format=None
        # )

        # self.val_gen = BSONIterator(train_bson_file, val_images_df, train_offsets_df,
        #                        self.num_classes, val_datagen, lock,
        #                        batch_size=self.batch_size, shuffle=True)


        self.model.fit_generator(
            generator=train_generator(),
            steps_per_epoch=np.ceil(nTrain / float(self.batch_size)),
            epochs=self.max_epochs,
            verbose=2,
            validation_data=val_generator(),
            validation_steps=np.ceil(nVal / float(self.batch_size)),
            callbacks=callbacks
        )

        # training
        # self.model.fit(X_train, y_train,
        #                validation_data = (X_test, y_test),
        #                epochs = self.max_epochs,
        #                verbose=1,
        #                shuffle = True,
        #                callbacks=callbacks)
    def test(self):
        self.model.load_weights('../weights/best_weights_{}.hdf5'.format(self.base_model))
        # make predictions
        test_predictions = self.model.predict(self.test_images)

        # save the predictions csv file
        pred_df = self.test_df[['id']].copy()
        pred_df['is_iceberg'] = test_predictions[:,1]
        pred_df.to_csv('../submit/predictions.csv', index = False)



if __name__ == '__main__':
    iceberg = Iceberg(base_model='simple')
    iceberg.train()
    iceberg.test()




