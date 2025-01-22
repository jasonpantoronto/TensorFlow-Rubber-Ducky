def load_and_preprocess_image(image_path):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    import numpy as np

    # Load the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    
    # Preprocess the image
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def augment_image(image):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    return datagen.flow(image, batch_size=1)

def resize_images(image_list, target_size=(224, 224)):
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    import numpy as np

    resized_images = []
    for img_path in image_list:
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img)
        resized_images.append(img_array)

    return np.array(resized_images)

def normalize_images(image_array):
    return image_array.astype('float32') / 255.0