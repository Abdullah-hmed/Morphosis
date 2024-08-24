from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def front_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        # print(request.files.keys())
        if 'file' not in request.files:
            return "no file received", 400
        image = request.files['file']
        if image.filename == '':
            return "no  file recived", 400
        if image:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)
            return "Thanks, file received successfully"
    return "Invalid Request", 400
    
# TODO: Display Image in DOM

if __name__ == "__main__":
    app.run(debug=True)
