from flask import Flask, render_template, request, redirect, send_from_directory, url_for, render_template_string
import os
import time
# AI Imports
# from StyleGAN_Pipeline.encode import ImageEncoder
# from StyleGAN_Pipeline.generate import ImageGenerator
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def front_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return '<div class="notification is-danger" remove-me="1s">No File Received</div>', 422
        image = request.files['file']
        if not is_valid_file_type(image.filename):
            return '<div class="notification is-danger" remove-me="1s">Invalid File Type</div>', 422
        if image.filename == '':
            return '<div class="notification is-danger" remove-me="1s">No File Received</div>', 422
        if image:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)
            time.sleep(1)
            return render_template_string("""<img src="{{ url_for('static', filename='uploads/' + filename) }}" alt="Uploaded image">""", filename=image.filename)
        return '<div class="notification is-danger" remove-me="1s">Invalid Request</div>', 422

def is_valid_file_type(file):
    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        return True
    else:
        return False

if __name__ == "__main__":
    app.run(debug=True)
