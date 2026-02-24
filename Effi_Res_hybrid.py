import tensorflow as tf
from keras import layers, models


def inception_block(x, filters):
    branch1x1 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(x)

    branch3x3 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    branch3x3 = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(branch3x3)

    branch5x5 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    branch5x5 = layers.Conv2D(filters, (5, 5), padding='same', activation='relu')(branch5x5)

    branch_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(branch_pool)

    x = layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1)
    x = layers.BatchNormalization()(x)
    return x


def mbconv_block(x, filters, kernel_size, strides=(1, 1), expand_ratio=6):
    input_tensor = x
    in_channels = x.shape[-1]

    x = layers.Conv2D(in_channels * expand_ratio, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if strides == (1, 1) and in_channels == filters:
        x = layers.add([x, input_tensor])

    return x


def residual_block(x, filters, strides=(1, 1)):
    shortcut = x

    x = layers.Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    if strides != (1, 1) or x.shape[-1] != shortcut.shape[-1]:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_feature_extractor(input_shape=(224, 224, 3), feature_dim=512):
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", use_bias=False, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 1 (no downsample): MBConv -> Res -> Inception
    x = mbconv_block(x, 64, (3, 3), strides=(1, 1), expand_ratio=1)
    x = residual_block(x, 64, strides=(1, 1))
    x = inception_block(x, 32)

    # Stage 2 (downsample): MBConv -> Res -> Inception
    x = mbconv_block(x, 128, (3, 3), strides=(2, 2), expand_ratio=6)
    x = residual_block(x, 128, strides=(1, 1))
    x = inception_block(x, 64)

    # Stage 3 (downsample): MBConv -> Res -> Inception
    x = mbconv_block(x, 256, (3, 3), strides=(2, 2), expand_ratio=6)
    x = residual_block(x, 256, strides=(1, 1))
    x = inception_block(x, 128)

    # Feature head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(feature_dim, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    return models.Model(inputs, x, name="effi_res_hybrid_feature_extractor")


if __name__ == "__main__":
    model = build_feature_extractor()
    model.summary()
