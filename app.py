from flask import Flask, render_template, request, send_from_directory
import os
import cv2  
from ultralytics import YOLO


model = YOLO(r'E:\data sciences\sharon\Road_Defects_detection\best_m.pt')

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = './saved_images'
app.config['OUTPUT_FOLDER'] = r'E:\data sciences\sharon\Road_Defects_detection\road_defects\predicted_images'



os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def detect():
    if request.method == "POST":
        # Save uploaded image
        image = request.files['image']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)
        
        
        img = cv2.imread(image_path)
        
       
        resized_img = cv2.resize(img, (680, 480))
        
         
        resized_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resized_' + image.filename)
        cv2.imwrite(resized_image_path, resized_img)

        pred = model.predict(resized_image_path)
        
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'predicted_' + image.filename)
        pred[0].save(output_path)

        return render_template('upload.html', prediction_image='predicted_' + image.filename)
    
    return render_template('upload.html')

@app.route('/predicted_images/<filename>')
def display_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
