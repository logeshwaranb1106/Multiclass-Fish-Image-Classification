import tensorflow as tf
from load_data import load_data
import os

train_data, val_data, _ = load_data()
IMG_SHAPE = (224, 224, 3)
EPOCHS = 10

MODELS = {
    "vgg16": tf.keras.applications.VGG16,
    "resnet50": tf.keras.applications.ResNet50,
    "mobilenetv2": tf.keras.applications.MobileNetV2,
    "inceptionv3": tf.keras.applications.InceptionV3,
    "efficientnetb0": tf.keras.applications.EfficientNetB0
}

def build_model(base_model_fn, input_shape, num_classes):
    base_model = base_model_fn(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze layers

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

os.makedirs("models", exist_ok=True)

for name, model_fn in MODELS.items():
    print(f"\nTraining {name.upper()}...")
    model = build_model(model_fn, IMG_SHAPE, train_data.num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

    model.save(f"models/{name}_fish_model.h5")
