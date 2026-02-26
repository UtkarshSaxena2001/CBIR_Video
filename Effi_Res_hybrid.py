import tensorflow as tf
from keras import layers, models  # type: ignore
from keras import ops as kops
from keras.applications import EfficientNetB4, InceptionResNetV2 # type: ignore


def _resize_to_match(reference, target):
    ref_shape = kops.shape(reference)
    target_h = ref_shape[1]
    target_w = ref_shape[2]
    return kops.image.resize(target, (target_h, target_w), interpolation="bilinear")


def _spatial_to_sequence(x):
    shape = kops.shape(x)
    batch = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]
    return kops.reshape(x, [batch, height * width, channels])


def build_feature_extractor(
    input_shape=(224, 224, 3),
    feature_dim=512,
    lstm_units=128,
    lstm_layers=5,
    spatial_size=7,
    efficientnet_weights=None,
    inception_resnet_weights=None,
):
    inputs = layers.Input(shape=input_shape)

    # EfficientNet B4 (v4 as requested)
    effi = EfficientNetB4(
        include_top=False,
        weights=efficientnet_weights,
        input_tensor=inputs,
    )
    effi_features = effi.output

    # Inception-ResNet (ResNet with Inception)
    inc_res = InceptionResNetV2(
        include_top=False,
        weights=inception_resnet_weights,
        input_tensor=inputs,
    )
    inc_features = inc_res.output

    # Align spatial sizes and concatenate along channels
    def _resize_output_shape(input_shape):
        return (
            input_shape[0],
            effi_features.shape[1],
            effi_features.shape[2],
            input_shape[3],
        )

    inc_features = layers.Lambda(
        lambda t: _resize_to_match(effi_features, t),
        output_shape=_resize_output_shape,
    )(inc_features)
    fused = layers.Concatenate(axis=-1)([effi_features, inc_features])

    # Reduce spatial resolution before LSTM to limit memory usage.
    if spatial_size is not None:
        fused = layers.Resizing(spatial_size, spatial_size, interpolation="bilinear")(fused)

    # Convert spatial map into a sequence (H*W timesteps) for LSTM
    seq = layers.Lambda(_spatial_to_sequence)(fused)

    # Stack LSTM layers (at least 5 as requested)
    x = seq
    for i in range(max(5, lstm_layers)):
        return_sequences = i < max(5, lstm_layers) - 1
        x = layers.LSTM(lstm_units, return_sequences=return_sequences, name=f"lstm_{i+1}")(x)

    # Feature head
    x = layers.Dense(feature_dim, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    return models.Model(inputs, x, name="effi_inceptionres_lstm_feature_extractor")


if __name__ == "__main__":
    model = build_feature_extractor()
    model.summary()
