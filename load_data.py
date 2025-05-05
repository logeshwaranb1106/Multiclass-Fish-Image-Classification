from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(image_size=(224, 224), batch_size=32):
    # Augment only training data
    train_gen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=30,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_test_gen = ImageDataGenerator(rescale=1.0/255)

    # Correct paths for the 'train', 'val', and 'test' directories
    base_path = r'L:\Guvi\Project 5\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data'

    # Load training data with augmentation
    train_data = train_gen.flow_from_directory(
        base_path + r'\train',  # Make sure this path points to the correct location
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Load validation data without augmentation
    val_data = val_test_gen.flow_from_directory(
        base_path + r'\val',  # Correct path to validation folder
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Load test data without augmentation
    test_data = val_test_gen.flow_from_directory(
        base_path + r'\test',  # Correct path to test folder
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_data, val_data, test_data

# Test (optional, for debugging)
if __name__ == "__main__":
    train_data, val_data, test_data = load_data()
    print("Train samples:", train_data.samples)
    print("Validation samples:", val_data.samples)
    print("Test samples:", test_data.samples)
    print("Class labels:", train_data.class_indices)
