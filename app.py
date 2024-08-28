from flask import Flask, render_template, request, redirect, send_from_directory, url_for, render_template_string, session
import os
import time
import secrets
from timeit import default_timer as timer
# AI Imports
from StyleGAN_Pipeline.encode import ImageEncoder, NoFaceDetectedException
from StyleGAN_Pipeline.generate import ImageGenerator
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = secrets.token_hex()



encoder = ImageEncoder()
generator = ImageGenerator()

def generateBaseImage():
    """
    Generates a base64 encoded image from an uploaded image. 
    
    This function takes no arguments and returns a base64 encoded string 
    representing an image generated from the uploaded image. The image is 
    first encoded using the ImageEncoder class, and then passed to the 
    ImageGenerator class to generate a new image. The generated image is then 
    encoded as a base64 string and returned.
    
    Returns:
        str: A base64 encoded string representing the generated image.
    """
    uploaded_image = session['image_path']
    print("Uploaded Image Path: ",upload_image)
    latent = encoder.encode_image(uploaded_image)
    generated_image_base64 = generator.generate_image(latent)
    return generated_image_base64

def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


def print_error_message(error_text):
    return f'<div class="notification is-danger" remove-me="1s">{error_text}</div>'


@app.route('/')
def front_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return print_error_message("No File Received"), 422
        image = request.files['file']
        if not is_valid_file_type(image.filename):
            return print_error_message("Invalid File Type"), 422
        if image.filename == '':
            return print_error_message("No File Received"), 422
        if image:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)
            session['image_path'] = image_path
            try:
                encode_start_time = timer()
                image_data = generateBaseImage()
                encode_end_time = timer()
                print_train_time(start=encode_start_time, end=encode_end_time, device="cuda")
            except NoFaceDetectedException:
                print("NoFaceDetectedException Occurred")
                return print_error_message("No Face Detected in Uploaded Image"), 422
            return render_template('editor.html', image=image_data)
        return print_error_message("Invalid Request"), 422


@app.route('/editor')
def editor_page():
    return render_template('editor.html')


def is_valid_file_type(file):
    print(file)
    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        print("valid file")
        return True
    else:
        print("invalid file")
        return False

if __name__ == "__main__":
    app.run(debug=True)
