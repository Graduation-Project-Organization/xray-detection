# import libaries
import os
import tensorflow as tf
from PIL import Image
import numpy as np

IMAGE_SIZE = (256, 256)
CLASS_LABELS = ['NORMAL', 'PNEUMONIA']

class Model:
    def __init__(self, model_path: str):
        # load the model
        self.model = tf.keras.models.load_model(model_path)

    def preprocess_image(self, image: Image.Image) -> np.array:
        # resize the image
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)
        
        # normalizes pixel values to [0, 1]
        img_array = np.array(image).astype(np.float32) / 255.0

        # expands dimensions to match model input shape
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image: Image.Image):
        # perform image preprocessing
        img_array = self.preprocess_image(image)

        # predict
        prediction = self.model.predict(img_array)[0][0]
        actual_prediction = int(prediction > 0.5)

        return CLASS_LABELS[actual_prediction]

    def close(self):
        del self.model

# test the class
#if __name__ == '__main__':
    #model = Model(os.path.join("assets", "model.h5"))
    #image = Image.open(os.path.join("assets", "image_path"))
    #print(model.classify(image))
