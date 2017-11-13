import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import KFold, StratifiedKFold

from models import Models
from helpers import transformations, flip, random_crop, expand_chan

flag_expand_chan = True

class Iceberg:
    def __init__(self, height=70, width=70, batch_size=128, max_epochs=500, base_model='simple', num_classes=2):
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.base_model = base_model
        self.num_classes = num_classes
        self.load_data()
        self.num_folds = 5

        self.define_model(self.test_images.shape[-1])

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


    def define_model(self, num_chan):
        models = Models(input_shape=(self.height, self.width, num_chan + 1*flag_expand_chan),
                        classes=self.num_classes)
        if self.base_model == 'vgg16':
            models.vgg16()
        elif self.base_model == 'vgg19':
            models.vgg19()
        elif self.base_model == 'resnet50':
            models.resnet50()
        elif self.base_model == 'inceptionV3':
            models.inceptionV3()
        elif self.base_model == 'xception':
            models.xception()
        elif self.base_model == 'simple':
            models.simple()  # TODO
        elif self.base_model == 'simple_inception':
            models.simple_resnet()
        elif self.base_model == 'pspnet':
            models.simple_pspnet()
        else:
            print('Uknown base model')
            raise SystemExit

        # models.compile(optimizer=RMSprop(lr=1e-3))

        models.compile(optimizer=Adam())

        self.model = models.get_model()

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
                                               patience=8,
                                               verbose=1,
                                               epsilon=1e-4),
                             EarlyStopping(min_delta=1e-4,
                                           patience=15,
                                           verbose=1)]



        nTrain = len(X_train)
        nVal = len(X_val)
        print("# training images: ", nTrain)
        print("# validation images: ", nVal)

        train_datagen = ImageDataGenerator(
            zca_whitening=True,
            shear_range=0.2,
            zoom_range=[1, 1.2],
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
                        x = X_train[i]
                        if flag_expand_chan:
                            x = expand_chan(x)
                        x = train_datagen.random_transform(x)
                        # x = transformations(x, np.random.randint(3))
                        x = random_crop(x, (self.height, self.width))
                        x_batch.append(x)
                        # x_batch.append(train_datagen.random_transform(X_train[i]))
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
                        x = X_val[i]
                        if flag_expand_chan:
                            x = expand_chan(x)
                        x = val_datagen.random_transform(x)
                        x = random_crop(x, (self.height, self.width), center=True)
                        x_batch.append(x)
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
            steps_per_epoch=5*np.ceil(nTrain / float(self.batch_size)),
            epochs=self.max_epochs,
            verbose=2,
            validation_data=val_generator(),
            validation_steps=5*np.ceil(nVal / float(self.batch_size)),
            callbacks=callbacks
        )

        # training
        # self.model.fit(X_train, y_train,
        #                validation_split=0.2,
        #                # validation_data = (X_val, y_val),
        #                epochs = self.max_epochs,
        #                verbose=2,
        #                shuffle = True,
        #                callbacks=callbacks)
    def test(self):
        # make predictions
        test_array_shape = self.test_images.shape # (8424, 75, 75, 2)
        aug_test_images = np.random.randn(test_array_shape[0], self.height, self.width, test_array_shape[-1] + flag_expand_chan)

        self.model.load_weights('../weights/best_weights_{}.hdf5'.format(self.base_model))

        nTest = len(self.test_images)
        K = 5
        print("# test images: ", nTest)
        test_predictions = 0
        for k in range(K):
            for i in range(nTest):
                x = self.test_images[i]
                if flag_expand_chan:
                    x = expand_chan(x)
                aug_test_images[i] = random_crop(x, (self.height, self.width))
                # test_predictions = self.model.predict(self.test_images)
            test_predictions += self.model.predict(aug_test_images) / float(K)


        # save the predictions csv file
        pred_df = self.test_df[['id']].copy()
        pred_df['is_iceberg'] = test_predictions[:,1]
        pred_df.to_csv('../submit/predictions.csv', index = False)



    def train_ensemble(self):  # split the train dataset 50%-50% (train-validation)
        train_labels = to_categorical(self.train_df['is_iceberg'])
        print(train_labels[:100, 1])

        # kf = KFold(len(self.train_images), n_folds=self.num_folds, shuffle=True, random_state=1)
        # kf = StratifiedKFold(train_labels, n_folds=self.num_folds, shuffle=True, random_state=1)
        kf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=1)

        fold_id = 0
        for train_index, val_index in kf.split(self.train_images, train_labels[:,1]):
            X_train = self.train_images[train_index]
            X_val = self.train_images[val_index]
            y_train = train_labels[train_index]
            y_val = train_labels[val_index]

            kfold_weights_path = '../weights/best_weights_{}_{}.hdf5'.format(self.base_model, fold_id)
            fold_id += 1

            print('Train', X_train.shape, y_train.shape)
            print('Validation', X_val.shape, y_val.shape)


            callbacks = [ModelCheckpoint(filepath=kfold_weights_path,
                                                 save_best_only=True,
                                                 save_weights_only=True),
                                 ReduceLROnPlateau(factor=0.5,
                                                   patience=4,
                                                   verbose=1,
                                                   epsilon=1e-4),
                                 EarlyStopping(min_delta=1e-4,
                                               patience=10,
                                               verbose=1)]

            nTrain = len(X_train)
            nVal = len(X_val)
            print("# training images: ", nTrain)
            print("# validation images: ", nVal)

            train_datagen = ImageDataGenerator(
                zca_whitening=True,
                shear_range=0.2,
                zoom_range=[1, 1.2],
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
                            x = X_train[i]
                            if flag_expand_chan:
                                x = expand_chan(x)
                            x = train_datagen.random_transform(x)
                            x = random_crop(x, (self.height, self.width))
                            # x = transformations(x, np.random.randint(3))
                            x_batch.append(x)
                            # x_batch.append(train_datagen.random_transform(X_train[i]))
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

            val_datagen = ImageDataGenerator(
                zca_whitening=True
            )
            def val_generator():
                while True:
                    for start in range(0, nVal, self.batch_size):
                        x_batch = []
                        y_batch = []
                        end = min(start + self.batch_size, nVal)

                        for i in range(start, end):
                            x = X_val[i]
                            if flag_expand_chan:
                                x = expand_chan(x)
                            x = val_datagen.random_transform(x)
                            x = random_crop(x, (self.height, self.width), center=True)
                            x_batch.append(x)
                            y_batch.append(y_val[i])
                        x_batch = np.array(x_batch, np.float32)
                        y_batch = np.array(y_batch, np.float32)
                        yield x_batch, y_batch

            self.model.fit_generator(
                generator=train_generator(),
                steps_per_epoch=5*np.ceil(nTrain / float(self.batch_size)),
                epochs=self.max_epochs,
                verbose=2,
                validation_data=val_generator(),
                validation_steps=5*np.ceil(nVal / float(self.batch_size)),
                callbacks=callbacks
            )


    def test_ensemble(self):
        test_array_shape = self.test_images.shape  # (8424, 75, 75, 2)
        preds = 0
        nTest = len(self.test_images)
        K = 5
        print(self.test_images.shape)
        print("# test images: ", nTest)
        test_predictions = 0
        for fold_id in range(self.num_folds):
            kfold_weights_path = '../weights/best_weights_{}_{}.hdf5'.format(self.base_model, fold_id)
            self.model.load_weights(kfold_weights_path)

            # make predictions with TTA

            # test_predictions = self.model.predict(self.test_images)
            # preds += test_predictions[:, 1] / float(K*self.num_folds)
            #
            # for k in range(K-1):
            #     aug_test_images = self.test_images
            #     for i in range(nTest):
            #         aug_test_images[i] = transformations(aug_test_images[i], k)
            #
            #     test_predictions = self.model.predict(aug_test_images)
            #     preds += test_predictions[:,1] / float(K*self.num_folds)


            aug_test_images = np.random.randn(test_array_shape[0], self.height, self.width, test_array_shape[-1] + 1*flag_expand_chan)
            for k in range(K):
                for i in range(nTest):
                    x = self.test_images[i]
                    if flag_expand_chan:
                        x = expand_chan(x)
                    aug_test_images[i] = random_crop(x, (self.height, self.width))
                    # test_predictions = self.model.predict(self.test_images)
                test_predictions += self.model.predict(aug_test_images) / float(K * self.num_folds)

        # save the predictions csv file
        pred_df = self.test_df[['id']].copy()
        pred_df['is_iceberg'] = np.clip(test_predictions[:, 1], 0, 1)
        pred_df.to_csv('../submit/predictions_ensemble.csv', index = False)



if __name__ == '__main__':
    iceberg = Iceberg(base_model='xception')
    # iceberg.train_ensemble()
    # iceberg.test_ensemble()

    iceberg.train()
    iceberg.test()



