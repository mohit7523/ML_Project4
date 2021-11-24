import os

import numpy as np
from PIL import Image
import tensorflow
from skimage import transform

from flask import Flask, flash, request, redirect, url_for, render_template
from flask import request
from werkzeug.utils import secure_filename


app = Flask(__name__)

# Change this model path according to project structure
MODEL_PATH = './model/model.hdf5'

model = tensorflow.keras.models.load_model(MODEL_PATH)


def model_predict(img_path, model):
    # Returns 0 for non-tb, 1 for normal and 2 for tb
    np_image = Image.open(img_path)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)

    pred = model.predict(np_image)

    if int(round(pred[0][0])) == 1:
        return (0, pred[0][0])
    elif int(round(pred[0][1])) == 1:
        return (1, pred[0][1])
    elif int(round(pred[0][2])) == 1:
        return (2, pred[0][2])


UPLOAD_FOLDER = 'static'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)


@app.route('/tb', methods=['POST'])
def tb():
    if request.method == 'POST':
        f = request.files['file']
        pred, prob = model_predict(f, model)
        prob = round(prob * 100, 2)
    return render_template('tb.html', pred=pred, prob=prob)


if __name__ == "__main__":
    app.run()
