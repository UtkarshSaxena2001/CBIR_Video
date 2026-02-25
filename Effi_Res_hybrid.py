import tensorflow as tf
from keras import layers, models #type: ignore


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


def stage_block(x, filters, mbconv_stride, expand_ratio):
    x = mbconv_block(x, filters, (3, 3), strides=mbconv_stride, expand_ratio=expand_ratio)
    x = residual_block(x, filters, strides=(1, 1))
    return x


def build_feature_extractor(input_shape=(224, 224, 3), feature_dim=512):
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", use_bias=False, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # EfficientNet + ResNet hybrid stages
    # Stage 1: keep resolution
    x = stage_block(x, filters=64, mbconv_stride=(1, 1), expand_ratio=1)
    # Stage 2: downsample
    x = stage_block(x, filters=128, mbconv_stride=(2, 2), expand_ratio=6)
    # Stage 3: downsample
    x = stage_block(x, filters=256, mbconv_stride=(2, 2), expand_ratio=6)
    # Stage 4: downsample
    x = stage_block(x, filters=512, mbconv_stride=(2, 2), expand_ratio=6)

    # Feature head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(feature_dim, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    return models.Model(inputs, x, name="effi_res_hybrid_feature_extractor")


if __name__ == "__main__":
    model = build_feature_extractor()
    model.summary()
