import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import RMSprop, Adam, SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import KFold, StratifiedKFold

from models import Models
from helpers import *
from loss import focus_loss
from keras import backend as K
from keras.losses import binary_crossentropy

flag_expand_chan = True

class Iceberg:
    def __init__(self, height=70, width=70, batch_size=128, max_epochs=500, base_model='simple', num_classes=2, use_inc_angle=False):
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.base_model = base_model
        self.num_classes = num_classes
        self.use_inc_angle = use_inc_angle
        self.define_model()


    def define_model(self):
        models = Models(input_shape=(self.height, self.width, INPUT_CHANNELS + 1*flag_expand_chan),
                        classes=self.num_classes, aux_input_shape = (1,))
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
        elif self.base_model == 'simple_resnet':
            models.simple_resnet()
        elif self.base_model == 'pspnet':
            models.simple_pspnet()
        elif self.base_model == 'inceptionResnet':
            models.inceptionResnetV2()
        else:
            print('Uknown base model')
            raise SystemExit

        # self.models = models

        # models.compile(optimizer=RMSprop(lr=1e-3))

        # models.compile(optimizer=Adam(lr=1e-3))
        sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        models.compile(optimizer=sgd)
        self.model = models.get_model()
        self.model.save_weights('../weights/init.hdf5')


    def train(self):
        # split the train dataset 50%-50% (train-validation)
        # load datasets
        train_df, train_images = load_and_format('../input/train.json', self.use_inc_angle)
        print('training', train_df.shape, 'loaded', [len(train_images), train_images[0][0].shape])

        train_labels = to_categorical(train_df['is_iceberg'])

        X_train_raw, X_val_raw, y_train, y_val = \
            train_test_split(train_images, train_labels, random_state=2017, test_size=0.2)

        X_train, inc_angle_train = zip(*X_train_raw)
        X_val, inc_angle_val = zip(*X_val_raw)

        X_train = np.array(X_train)
        X_val = np.array(X_val)

        print('Train: ', X_train.shape, y_train.shape)
        print('Validation: ', X_val.shape, y_val.shape)

        callbacks = [ModelCheckpoint(filepath='../weights/best_weights_{}_{}.hdf5'.format(self.base_model, self.use_inc_angle),
                                     monitor='val_loss',
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
            # featurewise_center=True,
            # featurewise_std_normalization=True,
            # zca_whitening=True,
            # shear_range=0.1,
            # zoom_range=[1, 1.1],
            # rotation_range=20,
            # horizontal_flip=True,
            # vertical_flip=True
        )

        def train_generator():
            while True:
                for start in range(0, nTrain, self.batch_size):
                    x_batch = []
                    y_batch = []
                    inc_angle_batch = []
                    end = min(start + self.batch_size, nTrain)

                    for i in range(start, end):
                        inc_angle_batch.append(inc_angle_train[i])
                        x = X_train[i]
                        if flag_expand_chan:
                            x = expand_chan(x)
                        x = train_datagen.random_transform(x)
                        # train_datagen.fit(x[np.newaxis,...])
                        # x = train_datagen.standardize(x)
                        # x = (x - np.mean(x, axis=(0,1))) / np.std(x, axis=(0,1))

                        x = random_crop(x, (self.height, self.width))
                        # x = random_rotation(x, np.random.randint(3))
                        x_batch.append(x)
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
                    if self.use_inc_angle:
                        inc_angle_batch = np.array(inc_angle_batch, np.float32)
                        yield [x_batch, inc_angle_batch], y_batch
                    else:
                        yield [x_batch], y_batch



        # train_gen = train_datagen.flow(
        #     X_train, y_train, shuffle=True, batch_size=self.batch_size
        # )
        # train_gen = train_datagen.flow_from_directory(
        #     self.train_images,
        #     target_size=(self.height, self.width),
        #     batch_size=self.batch_size,
        #     class_mode='binary'
        # )

        val_datagen = ImageDataGenerator(
            # featurewise_center=True,
            # featurewise_std_normalization=True,
            # horizontal_flip=True,
            # vertical_flip=True
        )
        def val_generator():
            while True:
                for start in range(0, nVal, self.batch_size): # TODO
                    x_batch = []
                    y_batch = []
                    inc_angle_batch = []

                    end = min(start + self.batch_size, nVal)

                    for i in range(start, end):
                        inc_angle_batch.append(inc_angle_val[i])
                        x = X_val[i]
                        if flag_expand_chan:
                            x = expand_chan(x)
                        x = val_datagen.random_transform(x)
                        # val_datagen.fit(x[np.newaxis,...])
                        # x = val_datagen.standardize(x)
                        # x = (x - np.mean(x, axis=(0,1))) / np.std(x, axis=(0,1))

                        x = random_crop(x, (self.height, self.width), center=True)
                        x_batch.append(x)
                        y_batch.append(y_val[i])
                    x_batch = np.array(x_batch, np.float32)
                    y_batch = np.array(y_batch, np.float32)

                    if self.use_inc_angle:
                        inc_angle_batch = np.array(inc_angle_batch, np.float32)
                        yield [x_batch, inc_angle_batch], y_batch
                    else:
                        yield [x_batch], y_batch


        # val_gen = val_datagen.flow(
        #     X_test, y_test, batch_size=self.batch_size, save_format=None
        # )

        self.model.fit_generator(
            generator=train_generator(),
            steps_per_epoch=5*np.ceil(nTrain / float(self.batch_size)),
            epochs=self.max_epochs,
            verbose=2,
            validation_data=val_generator(),
            validation_steps=5*np.ceil(nVal / float(self.batch_size)),
            callbacks=callbacks
        )

        # self.model.fit(X_train, y_train,
        #                validation_split=0.2,
        #                # validation_data = (X_val, y_val),
        #                epochs = self.max_epochs,
        #                verbose=2,
        #                shuffle = True,
        #                callbacks=callbacks)

        # compute val_loss
        y_pred = self.model.predict_generator(
            generator=val_generator(),
            steps=np.ceil(nVal / float(self.batch_size))
        )

        y_pred = K.variable(y_pred)
        y_val = K.variable(y_val)
        loss_i = K.eval(K.mean(binary_crossentropy(y_val, y_pred)))
        print("loss i: ", loss_i)

    def test(self):

        # load datasets
        test_df, test_images = load_and_format('../input/test.json', False)
        print('testing', test_df.shape, 'loaded', [len(test_images), test_images[0][0].shape])

        nTest = len(test_images)

        # make predictions
        test_array_shape = [nTest] + list(test_images[0][0].shape)  # (8424, 75, 75, 2)

        aug_test_images = np.random.randn(nTest, self.height, self.width, test_array_shape[-1] + flag_expand_chan)
        test_inc_angles = np.random.randn(nTest, 1)

        self.model.load_weights('../weights/best_weights_{}_{}.hdf5'.format(self.base_model, self.use_inc_angle))

        print("# test images: ", nTest)
        test_predictions = 0

        # test_datagen = ImageDataGenerator(
        #     featurewise_center=True,
        #     featurewise_std_normalization=True,
        #     horizontal_flip=True,
        #     vertical_flip=True
        # )

        for k in range(NUM_TTA):
            for i in range(nTest):
                x, inc_angle = test_images[i]
                if flag_expand_chan:
                    x = expand_chan(x)
                # test_datagen.fit(x[np.newaxis,...])
                # x = test_datagen.standardize(x)
                # x = (x - np.mean(x, axis=(0, 1))) / np.std(x, axis=(0, 1))

                aug_test_images[i] = fixed_crop(x, (self.height, self.width), k)
                # aug_test_images[i] = random_crop(x, (self.height, self.width))
                test_inc_angles[i] = inc_angle

            # test_predictions += self.model.predict([aug_test_images, test_inc_angles]) / float(NUM_TTA)
            test_predictions += self.model.predict([aug_test_images]) / float(NUM_TTA)


        # save the predictions csv file
        pred_df = test_df[['id']].copy()
        pred_df['is_iceberg'] = test_predictions[:,1]
        pred_df.to_csv('../submit/predictions.csv', index = False)



    def train_ensemble(self):
        # load datasets
        train_df, train_images = load_and_format('../input/train.json', self.use_inc_angle)
        print('training', train_df.shape, 'loaded', [len(train_images), train_images[0][0].shape])

        train_labels = to_categorical(train_df['is_iceberg'])
        print(train_labels[:100, 1])

        # kf = KFold(len(self.train_images), n_folds=NUM_CV_FOLDS, shuffle=True, random_state=1)
        # kf = StratifiedKFold(train_labels, n_folds=NUM_CV_FOLDS, shuffle=True, random_state=1)
        kf = StratifiedKFold(n_splits=NUM_CV_FOLDS, shuffle=True, random_state=1)

        loss = 0.0
        fold_id = 0
        for train_index, val_index in kf.split(train_images, train_labels[:,1]):
            # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
            # self.models.compile(optimizer=sgd)
            # model = self.models.get_model()

            model = self.model
            K.set_value(model.optimizer.lr, 1e-3)
            model.load_weights('../weights/init.hdf5')


            X_train, inc_angle_train = zip(*[train_images[i] for i in train_index])
            X_val, inc_angle_val = zip(*[train_images[i] for i in val_index])

            X_train = np.array(X_train)
            X_val = np.array(X_val)

            y_train = train_labels[train_index]
            y_val = train_labels[val_index]

            kfold_weights_path = '../weights/best_weights_{}_{}_{}.hdf5'.format(self.base_model, self.use_inc_angle, fold_id)
            # self.model.load_weights(kfold_weights_path)
            fold_id += 1

            print('Train: ', X_train.shape, y_train.shape)
            print('Validation: ', X_val.shape, y_val.shape)


            callbacks = [ModelCheckpoint(filepath=kfold_weights_path,
                                         monitor='val_loss',
                                         mode='min',
                                         save_best_only=True,
                                         save_weights_only=True),
                         ReduceLROnPlateau(factor=0.5,
                                           patience=4,
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
                # zca_whitening=True,
                # shear_range=0.1,
                # zoom_range=[1, 1.1],
                # rotation_range=20
                # horizontal_flip=True,
                # vertical_flip=True
            )

            def train_generator():
                while True:
                    for start in range(0, nTrain, self.batch_size):
                        x_batch = []
                        inc_angle_batch = []
                        y_batch = []
                        end = min(start + self.batch_size, nTrain)

                        for i in range(start, end):
                            inc_angle_batch.append(inc_angle_train[i])
                            x = X_train[i]
                            if flag_expand_chan:
                                x = expand_chan(x)
                            x = train_datagen.random_transform(x)
                            # x = (x - np.mean(x, axis=(0, 1))) / np.std(x, axis=(0, 1))

                            x = random_crop(x, (self.height, self.width))
                            # x = random_rotation(x, np.random.randint(3))
                            x_batch.append(x)
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

                        if self.use_inc_angle:
                            inc_angle_batch = np.array(inc_angle_batch, np.float32)
                            yield [x_batch, inc_angle_batch], y_batch
                        else:
                            yield [x_batch], y_batch


            val_datagen = ImageDataGenerator(
                zca_whitening=True,
                # horizontal_flip=True,
                # vertical_flip=True
            )
            def val_generator():
                while True:
                    for start in range(0, nVal, self.batch_size):
                        x_batch = []
                        inc_angle_batch = []
                        y_batch = []
                        end = min(start + self.batch_size, nVal)

                        for i in range(start, end):
                            inc_angle_batch.append(inc_angle_val[i])
                            x = X_val[i]
                            if flag_expand_chan:
                                x = expand_chan(x)
                            x = val_datagen.random_transform(x)
                            # x = (x - np.mean(x, axis=(0, 1))) / np.std(x, axis=(0, 1))

                            x = random_crop(x, (self.height, self.width), center=True)
                            x_batch.append(x)
                            y_batch.append(y_val[i])

                        x_batch = np.array(x_batch, np.float32)
                        y_batch = np.array(y_batch, np.float32)

                        if self.use_inc_angle:
                            inc_angle_batch = np.array(inc_angle_batch, np.float32)
                            yield [x_batch, inc_angle_batch], y_batch
                        else:
                            yield [x_batch], y_batch


            model.fit_generator(
                generator=train_generator(),
                steps_per_epoch=5*np.ceil(nTrain / float(self.batch_size)),
                initial_epoch=0,
                epochs=self.max_epochs,
                verbose=2,
                validation_data=val_generator(),
                validation_steps=5*np.ceil(nVal / float(self.batch_size)),
                callbacks=callbacks
            )


            # compute val_loss
            model.load_weights(kfold_weights_path)
            y_pred = model.predict_generator(
                generator=val_generator(),
                steps=np.ceil(nVal / float(self.batch_size))
            )

            y_pred = K.variable(y_pred)
            y_val = K.variable(y_val)
            loss_i = K.eval(K.mean(binary_crossentropy(y_val, y_pred))) / float(NUM_CV_FOLDS)
            print("loss i: ", loss_i)
            loss += loss_i
        print("loss: ", loss)

    def test_ensemble(self):
        # load datasets
        test_df, test_images = load_and_format('../input/test.json', False)
        print('testing', test_df.shape, 'loaded', [len(test_images), test_images[0][0].shape])

        preds = 0
        nTest = len(test_images)
        test_array_shape = [nTest] + list(test_images[0][0].shape)  # (8424, 75, 75, 2)
        print(test_array_shape)
        print("# test images: ", nTest)
        self.test_predictions = 0
        for fold_id in range(NUM_CV_FOLDS):
            kfold_weights_path = '../weights/best_weights_{}_{}_{}.hdf5'.format(self.base_model, self.use_inc_angle, fold_id)
            # model = self.models.get_model()
            model = self.model
            model.load_weights(kfold_weights_path)
            aug_test_images = np.random.randn(nTest, self.height, self.width, test_array_shape[-1] + 1*flag_expand_chan)
            test_inc_angles = np.random.randn(nTest, 1)

            for k in range(NUM_TTA):
                for i in range(nTest):
                    x, inc_angle = test_images[i]
                    if flag_expand_chan:
                        x = expand_chan(x)
                    # x = (x - np.mean(x, axis=(0, 1))) / np.std(x, axis=(0, 1))
                    # x = (x - np.min(x, axis=(0, 1))) / (np.max(x, axis=(0, 1)) - np.min(x, axis=(0, 1)))

                    aug_test_images[i] = fixed_crop(x, (self.height, self.width), k)
                    # aug_test_images[i] = random_crop(x, (self.height, self.width))
                    test_inc_angles[i] = inc_angle
                # self.test_predictions += model.predict([aug_test_images, test_inc_angles]) / float(NUM_TTA * NUM_CV_FOLDS)
                self.test_predictions += model.predict([aug_test_images]) / float(NUM_TTA * NUM_CV_FOLDS)


        # save the predictions csv file
        pred_df = test_df[['id']].copy()
        pred_df['is_iceberg'] = np.clip(self.test_predictions[:, 1], 0, 1)
        pred_df.to_csv('../submit/predictions_ensemble_{}_{}.csv'.format(self.base_model, self.use_inc_angle), index = False)



if __name__ == '__main__':

    if 1:
        pred = 0

        models = ['vgg16', 'resnet50', 'inceptionV3', 'simple', 'simple_resnet', 'pspnet', 'xception']
        # models = ['vgg16', 'resnet50', 'inceptionV3', 'simple', 'pspnet']

        iceberg = Iceberg()

        for base_model in models:
            print("------------------------------------------------------------------------------")
            print(base_model)

            iceberg.base_model = base_model
            iceberg.define_model()

            iceberg.train_ensemble()
            iceberg.test_ensemble()

            pred += iceberg.test_predictions / float(len(models))

            print("------------------------------------------------------------------------------")


        # pred_df = pd.read_csv('../submit/predictions_ensemble_vgg16.csv')
        # pred_df['is_iceberg'] = np.clip(pred[:, 1], 0, 1)
        # pred_df.to_csv('../submit/predictions_ensemble_{}_{}.csv'.format(len(models), self.use_inc_angle), index=False)

    else:
        iceberg = Iceberg(base_model='vgg16')
        iceberg.train_ensemble()
        iceberg.test_ensemble()

        # iceberg.train()
        # iceberg.test()
