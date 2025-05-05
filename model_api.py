# import dependencies
from flask import Flask, request, jsonify
from flask_cors import CORS
from model_core import Model
from PIL import Image
import os

# create a flask app
app = Flask(__name__)
# enable cors for all routes by default
CORS(app)

# load the model once the app starts
model_path = os.path.join("assets", "model.h5")
model = Model(model_path) 

# view api status
@app.route('/')
def index():
    return 'Image Classification API is running.'

# classify the x-ray
@app.route("/ai-xray-detection", methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    # load the image
    try:
        image_file = request.files['image']
        image = Image.open(image_file)
    except Exception:
        return jsonify({'error': 'Invalid image file'}), 400
    
    # predict and return result
    label = model.predict(image)
    return jsonify({'result': label})

if __name__ == '__main__':
    app.run()
