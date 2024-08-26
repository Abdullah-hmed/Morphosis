from flask import Flask, render_template, request, redirect, send_from_directory, url_for, render_template_string, session
import os
import time
import secrets
# AI Imports
# from StyleGAN_Pipeline.encode import ImageEncoder
# from StyleGAN_Pipeline.generate import ImageGenerator
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = secrets.token_hex()

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
            time.sleep(1)
            # return render_template_string("""<img src="{{ url_for('static', filename='uploads/' + filename) }}" alt="Uploaded image">""", filename=image.filename)
            return redirect('/editor')
        return print_error_message("Invalid Request"), 422


@app.route('/editor')
def editor_page():
    return render_template('editor.html', image=session['image_path'])


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
