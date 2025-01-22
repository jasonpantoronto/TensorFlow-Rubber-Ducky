import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load configuration
import yaml

with open('../config.yaml') as config_file:
    config = yaml.safe_load(config_file)

# Set parameters
batch_size = config['training']['batch_size']
img_height = config['training']['img_height']
img_width = config['training']['img_width']
train_data_dir = '../data/training/rubber_ducks'

# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=config['training']['rotation_range'],
    width_shift_range=config['training']['width_shift_range'],
    height_shift_range=config['training']['height_shift_range'],
    shear_range=config['training']['shear_range'],
    zoom_range=config['training']['zoom_range'],
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Load model
from model import create_model

model = create_model()

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=config['training']['epochs']
)

# Save the model
model.save('../model/rubber_duck_model.h5')