from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.regularizers import L1L2

from keras.layers import Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Conv2D, MaxPooling2D, \
    GlobalMaxPooling2D, Dropout, Input, Activation, UpSampling2D, AveragePooling2D
from keras.layers.merge import add, concatenate
from keras.models import Model

from loss import focus_loss

class Models:
    def __init__(self, input_shape, classes):
        self.input_shape = input_shape
        self.classes = classes
        self.model = Sequential()

    def vgg16(self):
        self.model.add(BatchNormalization(input_shape=self.input_shape))
        if not self.input_shape[-1] == 3:
            self.model.add(Conv2D(3, kernel_size=(1,1), padding='same'))

        import myVgg16
        base_model = myVgg16.VGG16(include_top=False, weights=None,
                                   input_shape=self.input_shape)

        self.model.add(base_model)
        self.model.add(Flatten())
        self.model.add(Dense(self.classes, activation='softmax'))

    def vgg19(self):
        base_model = VGG19(include_top=False, weights='imagenet',
                           input_shape=self.input_shape)

        self.model.add(base_model)
        self.model.add(Flatten())
        self.model.add(Dense(self.classes, activation='softmax'))

    def resnet50(self):
        self.model.add(BatchNormalization(input_shape=self.input_shape))
        import myResnet50
        base_model = myResnet50.ResNet50(include_top=False, weights=None,
                                         input_shape=self.input_shape)

        self.model.add(base_model)
        self.model.add(Flatten())
        self.model.add(Dense(self.classes, activation='softmax'))

    def inceptionV3(self):
        self.model.add(BatchNormalization(input_shape=self.input_shape))
        import myInception
        base_model = myInception.InceptionV3(include_top=False, weights=None,
                                             input_shape=self.input_shape)

        self.model.add(base_model)
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(self.classes, activation='softmax'))

    def xception(self):
        self.model.add(BatchNormalization(input_shape=self.input_shape))
        import myXception
        base_model = myXception.Xception(include_top=False, weights=None,
                                         input_shape=self.input_shape)
        self.model.add(base_model)
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(self.classes, activation='softmax'))

    def simple(self):
        # simple CNN
        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=self.input_shape))
        for i in range(4):
            self.model.add(Conv2D(16 * 2 ** i, kernel_size=(3, 3), padding='same'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))

            self.model.add(Conv2D(16 * 2 ** i, kernel_size=(3, 3), padding='same'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))

            if i < 4:
                self.model.add(MaxPooling2D((2, 2)))
        self.model.add(GlobalMaxPooling2D())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(16))
        self.model.add(Dense(2, activation='softmax'))

    def simple_resnet(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
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
            x = MaxPooling2D((2,2))(z)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(2, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=x)

    def simple_pspnet(self):
        from pspnet import pspnet2
        base_model = pspnet2(input_shape=self.input_shape)

        self.model.add(base_model)
        self.model.add(Flatten())
        self.model.add(Dense(self.classes, activation='softmax'))

    def simple_cascade_atrous(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        for i in range(2):
            x = Conv2D(8 * 2 ** i, kernel_size=(3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation(activation='relu')(x)
            x = MaxPooling2D((2,2))(x)

        for i in range(3):
            x = Conv2D(32, kernel_size=(3, 3), dilation_rate=2**(i+1), padding='same')(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(2, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=x)

    def simple_parallel_atrous(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        for i in range(2):
            x = Conv2D(8 * 2 ** i, kernel_size=(3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation(activation='relu')(x)
            x = MaxPooling2D((2,2))(x)

        # ASPP
        x1 = Conv2D(16, kernel_size=(1, 1), padding='same')(x)
        x1 = BatchNormalization()(x1)
        x1 = Activation(activation='relu')(x1)

        x2 = Conv2D(16, kernel_size=(3, 3), dilation_rate=2, padding='same')(x)
        x2 = BatchNormalization()(x2)
        x2 = Activation(activation='relu')(x2)

        x3 = Conv2D(16, kernel_size=(3, 3), dilation_rate=3, padding='same')(x)
        x3 = BatchNormalization()(x3)
        x3 = Activation(activation='relu')(x3)

        x4 = Conv2D(16, kernel_size=(3, 3), dilation_rate=6, padding='same')(x)
        x4 = BatchNormalization()(x4)
        x4 = Activation(activation='relu')(x4)

        x5 = AveragePooling2D(pool_size=(17, 17))(x)
        x5 = UpSampling2D(size=(17, 17))(x5)

        x = concatenate([x1, x2, x3, x4, x5])
        x = Conv2D(32, kernel_size=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(2, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=x)

    def compile(self, optimizer):
        print(self.model.summary())
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['accuracy', focus_loss])

    def load_weights(self, path):
        self.model.load_weights(path)

    def save_weights(self, path):
        self.model.save_weights(path)

    def get_model(self):
        return self.model