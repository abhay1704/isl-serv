from flask import Flask
from flask_cors import CORS
from flask import request
from utils import load_model, process, predict_1

app = Flask(__name__)


__model__ = None
__idx_to_class__ = None

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return 'No image file part in the request', 400

    try:
        # Save the image file in the image-folder
        image_file = request.files['image']
        image_tensor = process(image_file)
        if image_tensor is None:
            return 'No hands detected in the image', 400

        pred_prob, y = predict_1(__model__, image_tensor)
        pred_class = [__idx_to_class__[i] for i in y]
        return {'class': pred_class, 'prob': pred_prob}

    except Exception as e:
        return str(e), 400


if __name__ == '__main__':
    # Enable CORS for all routes
    CORS(app)
    __model__, __idx_to_class__ = load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
    