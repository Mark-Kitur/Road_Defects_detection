from flask import Flask, render_template, request, send_from_directory
import os
import cv2  # Import OpenCV for image resizing
from ultralytics import YOLO

# Load YOLO model
model = YOLO('best (1).pt')

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = './saved_images'
app.config['OUTPUT_FOLDER'] = './predicted_images'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def detect():
    if request.method == "POST":
        # Save uploaded image
        image = request.files['image']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)
        
        # Load the image using OpenCV
        img = cv2.imread(image_path)
        
        # Resize the image to 680x480
        resized_img = cv2.resize(img, (680, 480))
        
        # Save the resized image back to the same path (or a temporary path if preferred)
        resized_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resized_' + image.filename)
        cv2.imwrite(resized_image_path, resized_img)

        # Run prediction on the resized image
        pred = model.predict(resized_image_path)
        
        # Save the result with overlays to output folder
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'predicted_' + image.filename)
        pred[0].save(output_path)

        # Display the image with predictions
        return render_template('upload.html', prediction_image='predicted_' + image.filename)
    
    return render_template('upload.html')

# Route to display predicted image
@app.route('/predicted_images/<filename>')
def display_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
