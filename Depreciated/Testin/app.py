from flask import Flask, request
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'  
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'} 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Image Upload</title>
        </head>
        <body>
            <h1>Upload an Image for Analysis</h1>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required>
                <button type="submit">Analyze</button>
            </form>
        </body>
        </html>
    '''

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part"
    
    file = request.files['image']
    
    if file.filename == '':
        return "No selected file"
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)  # Save the file
        
        print(f"Uploaded file saved to: {filepath}")  # Debugging output
        
        result = predict(filepath)  
        
        return f"<h1>Prediction Result: {result}</h1>"
    
    return "File type not allowed"

def predict(image_path):
    # Here, implement your model prediction logic
    # For now, we will return a mock result to verify the flow
    print(f"Predicting for image: {image_path}")  # Debugging output
    # Normally, load your model and preprocess the image here
    return "Sample Prediction Result for " + os.path.basename(image_path)

if __name__ == '__main__':
    app.run(debug=True)
