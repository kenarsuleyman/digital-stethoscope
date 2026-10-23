"""
Model Architecture — MobileNetV2 Transfer Learning

Rescaling([0, 1] -> [-1, 1], the input range MobileNetV2's ImageNet weights expect)
→ MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
→ GlobalAveragePooling2D
→ Dropout(0.3)
→ Dense(1, activation='sigmoid')

Two functions:
- build_stage1_model(): All base layers frozen (feature extraction)
- unfreeze_for_stage2(model, n_blocks=4): Unfreeze last N inverted residual blocks
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2


def build_stage1_model(input_shape=(224, 224, 3)) -> Model:
    """
    Build MobileNetV2 with frozen base for Stage 1 training.

    Returns a compiled-ready model (compile separately to set optimizer/lr).
    """
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    # Mel images are min-max scaled to [0, 1]; scaling inside the model keeps the app's preprocessing unchanged
    x = layers.Rescaling(2.0, offset=-1.0, name="to_mobilenet_range")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="heart_sound_classifier")
    return model


def unfreeze_for_stage2(model: Model, n_blocks: int = 4) -> Model:
    """
    Unfreeze the last N inverted residual blocks of MobileNetV2 for fine-tuning.

    MobileNetV2 has 17 inverted residual blocks (block_1 through block_16 + the
    final conv). This function unfreezes from block_(16-n_blocks+1) onward.
    """
    base = next(l for l in model.layers if l.name.startswith("mobilenetv2"))
    base.trainable = True

    # Find the block to start unfreezing from
    unfreeze_from = f"block_{17 - n_blocks}_"

    found_unfreeze = False
    for layer in base.layers:
        if unfreeze_from in layer.name:
            found_unfreeze = True
        if not found_unfreeze:
            layer.trainable = False
        # BatchNorm layers should stay frozen even when unfreezing
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Count trainable vs frozen
    trainable = sum(1 for l in base.layers if l.trainable)
    frozen = sum(1 for l in base.layers if not l.trainable)

    print(f"Fine-tuning: {trainable} trainable layers, {frozen} frozen layers")
    print(f"Unfreezing from: {unfreeze_from}")

    return model


def get_model_summary(model: Model) -> str:
    """Return model summary as string."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)
