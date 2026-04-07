import tensorflow as tf
from tensorflow.keras import layers, models, applications
import config

def get_model(model_name, num_classes):
    # 1. Define the actual grayscale input shape
    img_input = layers.Input(shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 1))

    # 2. Convert 1-channel grayscale to 3-channel RGB 
    # This repeats the grayscale channel 3 times so InceptionResNetV2 can process it
    x = layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1))(img_input)

    # 3. Initialize the Inception engine
    if model_name == 'inception_resnet_v2':
        base_model = applications.InceptionResNetV2(
            weights=None,  # As per research paper: training from scratch
            include_top=False,
            input_tensor=x
        )
    
    # 4. Add the 'Drop-out Induced' Head from the Research Paper
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(config.DROPOUT_RATES[0])(x) # 0.3 Dropout
    
    # x = layers.Dense(512, activation='relu')(x)
    # x = layers.Dropout(config.DROPOUT_RATES[1])(x) # 0.2 Dropout
    
    x = layers.Dense(256, activation='relu')(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs=img_input, outputs=predictions)