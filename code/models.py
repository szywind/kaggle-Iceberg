from keras.applications import VGG16, ResNet50, InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Conv2D, MaxPooling2D, \
    Dropout, Input, Activation, Flatten
from keras.layers.merge import add, concatenate
from keras.models import Model

from loss import focus_loss
import myResnet50, myVgg16, myXception, myInception, myDensenet, myInceptionResnet
from pspnet import pspnet2
from constant import *


class Models:
    def __init__(self, input_shape, classes, aux_input_shape):
        self.input_shape = input_shape
        self.classes = classes
        self.aux_input_shape = aux_input_shape

    def model_factory(self, base_model, mode=2, dropout_rate=0.5):
        input = Input(shape=self.input_shape, name='main_input')
        # aux_input = Input(shape=self.aux_input_shape, name='aux_input')

        x = BatchNormalization()(input)
        if not self.input_shape[-1] == 3:
            x = Conv2D(3, kernel_size=(1, 1), padding='same', activation='relu')(x)
        x = base_model(x)
        if mode == 0:
            x = Flatten()(x)

        # x = Dropout(dropout_rate)(x)
        # x = Dense(16, activation='relu')(x)

        # aux_input_1 = Dense(1)(aux_input)
        # x = concatenate([x, aux_input_1])
        # x = Dense(16, activation='relu')(x)
        # x = Dropout(dropout_rate)(x)
        # aux_input_2 = Dense(1)(aux_input)
        # x = concatenate([x, aux_input_2])

        x = Dense(self.classes, activation='softmax')(x)
        # self.model = Model(inputs=[input, aux_input], outputs=x)
        self.model = Model(inputs=[input], outputs=x)


    def vgg16(self):
        base_model = myVgg16.VGG16(include_top=False, weights=None, input_shape=self.input_shape, pooling='avg')

        # base_model = VGG16(include_top=False, weights=None, input_shape=self.input_shape, pooling='avg')
        # base_model.load_weights('../imagenet_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
        # base_model.trainable = True
        self.model_factory(base_model)


    def resnet50(self):
        base_model = myResnet50.ResNet50(include_top=False, weights=None, input_shape=self.input_shape, pooling='avg')

        # base_model = ResNet50(include_top=False, weights=None, input_shape=self.input_shape, pooling='avg')
        # base_model.load_weights('../imagenet_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
        # base_model.trainable = True
        self.model_factory(base_model)

    def inceptionResnetV2(self):
        base_model = InceptionResNetV2(include_top=False, weights=None, input_shape=self.input_shape, pooling='avg')
        # base_model.load_weights('../imagenet_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
        # base_model.trainable = True
        self.model_factory(base_model)

    def inceptionV3(self):
        base_model = myInception.InceptionV3(include_top=False, weights=None, input_shape=self.input_shape, pooling='avg')
        self.model_factory(base_model)

    def xception(self):
        base_model = myXception.Xception(include_top=False, weights=None, input_shape=self.input_shape, pooling='avg')
        self.model_factory(base_model)

    def simple(self):
        def mySimple(input_shape):
            # simple CNN
            input = Input(shape=input_shape)
            x = input
            for i in range(4):
                x = Conv2D(16 * 2 ** i, kernel_size=(3, 3), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = Conv2D(16 * 2 ** i, kernel_size=(3, 3), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                x = MaxPooling2D(2, 2)(x)

            # x = GlobalAveragePooling2D()(x)
            x = GlobalMaxPooling2D()(x)

            return Model(input, x)

        base_model = mySimple(input_shape=self.input_shape)
        self.model_factory(base_model, mode=2)

    def simple_resnet(self):
        def mySimpleResnet(input_shape):
            input = Input(shape=input_shape)
            x = input
            for i in range(3):
                x = Conv2D(8 * 2 ** i, kernel_size=(3, 3), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation(activation='relu')(x)
                z = Conv2D(8 * 2 ** i, kernel_size=(3, 3), padding='same')(x)
                z = BatchNormalization()(z)
                z = Activation(activation='relu')(z)
                z = Conv2D(8 * 2 ** i, kernel_size=(3, 3), padding='same')(z)
                z = BatchNormalization()(z)
                z = add([x, z])
                z = Activation('relu')(z)

                x = MaxPooling2D((2, 2))(z)

            # x = GlobalAveragePooling2D()(x)
            x = GlobalMaxPooling2D()(x)

            return Model(input, x)

        base_model = mySimpleResnet(input_shape=self.input_shape)
        self.model_factory(base_model, mode=2)

    def simple_pspnet(self):
        base_model = pspnet2(input_shape=self.input_shape)
        self.model_factory(base_model)

    def compile(self, optimizer):
        print(self.model.summary())
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['accuracy', focus_loss])

    def load_weights(self, path):
        self.model.load_weights(path)

    def save_weights(self, path):
        self.model.save_weights(path)

    def get_model(self):
        return self.model