import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import base64
from io import BytesIO
matplotlib.pyplot.switch_backend('Agg') 

allowed_exts = {'jpg', 'jpeg','png','JPG','JPEG','PNG'}
app = Flask(__name__)

def check_allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

def edge_mask(img, line_size=7, blur_value=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges

def color_quantization(img, k=9):
    # Transform the image
    data = np.float32(img).reshape((-1, 3))
    # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    # Implementing K-Means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

def animate(filename, output_filename):
    img = cv2.imread(filename, -1)
    edges = edge_mask(img)
    quantized_img = color_quantization(img)
    blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200,sigmaSpace=200)
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
    output = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    plt.imshow(output)
    plt.imsave(output_filename, output)
    return output_filename

@app.route("/",methods=['GET', 'POST'])
def index():
	root = os.getcwd()
	if request.method == 'POST':
		if 'file' not in request.files:
			print('No file attached in request')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			print('No file selected')
			return redirect(request.url)
		if file and check_allowed_file(file.filename):
			input_filepath = os.path.join(root,secure_filename(file.filename))
			output_filepath = os.path.join(root, 'output.jpg')
			file.save(input_filepath)
			animate(input_filepath, output_filepath)
			with open(output_filepath, "rb") as image_file:
				encoded_string = base64.b64encode(image_file.read()).decode()
		return render_template('index.html', img_data=encoded_string), 200
	else:
		return render_template('index.html', img_data=""), 200

if __name__ == "__main__":
	app.debug=True
	app.run(host='0.0.0.0', debug=True)