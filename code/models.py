from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.regularizers import L1L2

from keras.layers import Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Conv2D, MaxPooling2D, \
    GlobalMaxPooling2D, Dropout, Input, Activation
from keras.layers.merge import add
from keras.models import Model

from loss import focus_loss

class Models:
    def __init__(self, input_shape, classes):
        self.input_shape = input_shape
        self.classes = classes
        self.model = Sequential()

    def vgg16(self):
        if not self.input_shape[-1] == 3:
            self.model = Sequential()
            self.model.add(BatchNormalization(input_shape=self.input_shape))
            self.model.add(Conv2D(3, kernel_size=(1,1), padding='same'))
        base_model = VGG16(include_top=False, weights='imagenet',
                           input_shape=[self.input_shape[0],self.input_shape[1],3])

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
        base_model = ResNet50(include_top=False, weights='imagenet',
                              input_shape=self.input_shape)

        self.model.add(base_model)
        self.model.add(Flatten())
        self.model.add(Dense(self.classes, activation='softmax'))

    def inceptionV3(self):
        base_model = InceptionV3(include_top=False, weights='imagenet',
                                 input_shape=self.input_shape)

        self.model.add(base_model)
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(self.classes, activation='softmax'))

    def xception(self):
        base_model = Xception(include_top=False, weights='imagenet',
                                 input_shape=self.input_shape)
        self.model.add(base_model)
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(self.classes, activation='softmax'))

    def simple(self):
        # simple CNN
        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=self.input_shape))
        for i in range(4):
            self.model.add(Conv2D(16 * 2 ** i, kernel_size=(3, 3), padding='same', activation='relu'))
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

    def compile(self, optimizer):
        print(self.model.summary())
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['accuracy', focus_loss])

    def load_weights(self, path):
        self.model.load_weights(path)

    def save_weights(self, path):
        self.model.save_weights(path)

    def get_model(self):
        return self.model