from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.regularizers import L1L2
from keras import backend as K
from keras.layers import AveragePooling2D, concatenate

from keras.layers import Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Conv2D, MaxPooling2D, \
    GlobalMaxPooling2D, Dropout, Input, Activation
from keras.layers.merge import add
from keras.models import Model

from loss import focus_loss


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

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


    def simple_inception(self):
        def InceptionV3(include_top=True,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=None,
                        pooling=None,
                        classes=1000,
                        factor=4):
            # Determine proper input shape

            img_input = Input(shape=self.input_shape)

            if K.image_data_format() == 'channels_first':
                channel_axis = 1
            else:
                channel_axis = 3

            x = conv2d_bn(img_input, 32//factor, 3, 3, strides=(2, 2), padding='valid')
            x = conv2d_bn(x, 32//factor, 3, 3, padding='valid')
            x = conv2d_bn(x, 64//factor, 3, 3)
            x = MaxPooling2D((3, 3), strides=(2, 2))(x)

            x = conv2d_bn(x, 80//factor, 1, 1, padding='valid')
            x = conv2d_bn(x, 192//factor, 3, 3, padding='valid')
            x = MaxPooling2D((3, 3), strides=(2, 2))(x)

            # mixed 0, 1, 2: 35 x 35 x 256
            branch1x1 = conv2d_bn(x, 64//factor, 1, 1)

            branch5x5 = conv2d_bn(x, 48//factor, 1, 1)
            branch5x5 = conv2d_bn(branch5x5, 64//factor, 5, 5)

            branch3x3dbl = conv2d_bn(x, 64//factor, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96//factor, 3, 3)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96//factor, 3, 3)

            branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 32//factor, 1, 1)
            x = concatenate(
                [branch1x1, branch5x5, branch3x3dbl, branch_pool],
                axis=channel_axis,
                name='mixed0')

            # mixed 1: 35 x 35 x 256
            branch1x1 = conv2d_bn(x, 64//factor, 1, 1)

            branch5x5 = conv2d_bn(x, 48//factor, 1, 1)
            branch5x5 = conv2d_bn(branch5x5, 64//factor, 5, 5)

            branch3x3dbl = conv2d_bn(x, 64//factor, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96//factor, 3, 3)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96//factor, 3, 3)

            branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 64//factor, 1, 1)
            x = concatenate(
                [branch1x1, branch5x5, branch3x3dbl, branch_pool],
                axis=channel_axis,
                name='mixed1')

            # mixed 2: 35 x 35 x 256
            branch1x1 = conv2d_bn(x, 64//factor, 1, 1)

            branch5x5 = conv2d_bn(x, 48//factor, 1, 1)
            branch5x5 = conv2d_bn(branch5x5, 64//factor, 5, 5)

            branch3x3dbl = conv2d_bn(x, 64//factor, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96//factor, 3, 3)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96//factor, 3, 3)

            branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 64//factor, 1, 1)
            x = concatenate(
                [branch1x1, branch5x5, branch3x3dbl, branch_pool],
                axis=channel_axis,
                name='mixed2')

            # mixed 3: 17 x 17 x 768
            branch3x3 = conv2d_bn(x, 384//factor, 3, 3, strides=(2, 2), padding='valid')

            branch3x3dbl = conv2d_bn(x, 64//factor, 1, 1)
            branch3x3dbl = conv2d_bn(branch3x3dbl, 96//factor, 3, 3)
            branch3x3dbl = conv2d_bn(
                branch3x3dbl, 96//factor, 3, 3, strides=(2, 2), padding='valid')

            branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
            x = concatenate(
                [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

            # mixed 4: 17 x 17 x 768
            branch1x1 = conv2d_bn(x, 192//factor, 1, 1)

            branch7x7 = conv2d_bn(x, 128//factor, 1, 1)
            branch7x7 = conv2d_bn(branch7x7, 128//factor, 1, 7)
            branch7x7 = conv2d_bn(branch7x7, 192//factor, 7, 1)

            branch7x7dbl = conv2d_bn(x, 128//factor, 1, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 128//factor, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 128//factor, 1, 7)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 128//factor, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 192//factor, 1, 7)

            branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 192//factor, 1, 1)
            x = concatenate(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=channel_axis,
                name='mixed4')

            # # mixed 5, 6: 17 x 17 x 768
            # for i in range(2):
            #     branch1x1 = conv2d_bn(x, 192, 1, 1)
            #
            #     branch7x7 = conv2d_bn(x, 160, 1, 1)
            #     branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
            #     branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
            #
            #     branch7x7dbl = conv2d_bn(x, 160, 1, 1)
            #     branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
            #     branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
            #     branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
            #     branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
            #
            #     branch_pool = AveragePooling2D(
            #         (3, 3), strides=(1, 1), padding='same')(x)
            #     branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            #     x = concatenate(
            #         [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            #         axis=channel_axis,
            #         name='mixed' + str(5 + i))
            #
            # # mixed 7: 17 x 17 x 768
            # branch1x1 = conv2d_bn(x, 192, 1, 1)
            #
            # branch7x7 = conv2d_bn(x, 192, 1, 1)
            # branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
            # branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
            #
            # branch7x7dbl = conv2d_bn(x, 192, 1, 1)
            # branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
            # branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
            # branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
            # branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
            #
            # branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
            # branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            # x = concatenate(
            #     [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            #     axis=channel_axis,
            #     name='mixed7')
            #
            # # mixed 8: 8 x 8 x 1280
            # branch3x3 = conv2d_bn(x, 192, 1, 1)
            # branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
            #                       strides=(2, 2), padding='valid')
            #
            # branch7x7x3 = conv2d_bn(x, 192, 1, 1)
            # branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
            # branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
            # branch7x7x3 = conv2d_bn(
            #     branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')
            #
            # branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
            # x = concatenate(
            #     [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')
            #
            # # mixed 9: 8 x 8 x 2048
            # for i in range(2):
            #     branch1x1 = conv2d_bn(x, 320, 1, 1)
            #
            #     branch3x3 = conv2d_bn(x, 384, 1, 1)
            #     branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
            #     branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
            #     branch3x3 = concatenate(
            #         [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))
            #
            #     branch3x3dbl = conv2d_bn(x, 448, 1, 1)
            #     branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
            #     branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
            #     branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
            #     branch3x3dbl = concatenate(
            #         [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)
            #
            #     branch_pool = AveragePooling2D(
            #         (3, 3), strides=(1, 1), padding='same')(x)
            #     branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            #     x = concatenate(
            #         [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            #         axis=channel_axis,
            #         name='mixed' + str(9 + i))
            if include_top:
                # Classification block
                x = GlobalAveragePooling2D(name='avg_pool')(x)
                x = Dense(classes, activation='softmax', name='predictions')(x)
            else:
                if pooling == 'avg':
                    x = GlobalAveragePooling2D()(x)
                elif pooling == 'max':
                    x = GlobalMaxPooling2D()(x)

            inputs = img_input
            # Create model.
            return Model(inputs, x, name='inception_v3')


        # self.model.add(BatchNormalization(input_shape=self.input_shape))

        base_model = InceptionV3(include_top=True, weights=None,
                                 input_shape=self.input_shape, classes=self.classes)
        self.model.add(base_model)
        # self.model.add(GlobalAveragePooling2D())
        # self.model.add(Dense(self.classes, activation='softmax'))

    def compile(self, optimizer):
        print(self.model.summary())
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['accuracy', focus_loss])

    def load_weights(self, path):
        self.model.load_weights(path)

    def save_weights(self, path):
        self.model.save_weights(path)

    def get_model(self):
        return self.model