from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.layers import Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Conv2D, MaxPooling2D, \
    GlobalMaxPooling2D, Dropout


class Models:
    def __init__(self, input_shape, classes):
        self.input_shape = input_shape
        self.classes = classes
        self.model = Sequential()

    def vgg16(self):
        base_model = VGG16(include_top=False, weights='imagenet',
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
        self.model.add(BatchNormalization(input_shape=(75, 75, 2)))
        for i in range(4):
            self.model.add(Conv2D(8 * 2 ** i, kernel_size=(3, 3)))
            self.model.add(MaxPooling2D((2, 2)))
        self.model.add(GlobalMaxPooling2D())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(8))
        self.model.add(Dense(2, activation='softmax'))

    def compile(self, optimizer):
        print(self.model.summary())
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['accuracy'])

    def load_weights(self, path):
        self.model.load_weights(path)

    def save_weights(self, path):
        self.model.save_weights(path)

    def get_model(self):
        return self.model