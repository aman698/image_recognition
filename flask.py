import os
import cv2
import face_recognition
import numpy as np
import pickle
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import relationship
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Configure the app and database
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'Images')  # Ensure
app.secret_key = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Database Models
class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    emp = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    department = db.Column(db.String(100), nullable=False)
    gmail = db.Column(db.String(100), nullable=False)
    star = db.Column(db.Integer, nullable=False)
    images = relationship('Image', backref='student', lazy=True)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(100), nullable=False)
    image_type = db.Column(db.String(50), nullable=False)  # e.g., 'front', 'left', 'back'
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    starttime = db.Column(db.DateTime, default=datetime.utcnow)
    endtime = db.Column(db.DateTime)
    student = db.relationship('Student', backref=db.backref('attendances', lazy=True))

# Create the database tables
with app.app_context():
    db.create_all()

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    students = Student.query.all()
    return render_template('upload.html', students=students)

@app.route('/upload', methods=['POST'])
def upload_image():
    # Check for required form data
    required_fields = ['name', 'emp', 'age', 'department', 'gmail', 'star']
    for field in required_fields:
        if field not in request.form or not request.form[field].strip():
            flash(f'Missing field: {field}')
            return redirect(request.url)
    
    # Check for image files
    required_images = ['front_image', 'left_image', 'back_image']
    for img in required_images:
        if img not in request.files or request.files[img].filename == '':
            flash(f'Missing image: {img.replace("_", " ").capitalize()}')
            return redirect(request.url)
    
    name = request.form['name'].strip()
    emp = request.form['emp'].strip()
    age = request.form['age'].strip()
    department = request.form['department'].strip()
    gmail = request.form['gmail'].strip()
    star = request.form['star'].strip()

    # Validate age
    try:
        age = int(age)
        if age <= 0:
            raise ValueError
    except ValueError:
        flash('Age must be a positive number')
        return redirect(request.url)

    # Create a new student
    try:
        new_student = Student(name=name, emp=emp, age=age, department=department, gmail=gmail, star=star)
        db.session.add(new_student)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        flash(f'Error saving student to database: {e}')
        return redirect(request.url)
    
    # Save each image
    for img_type in required_images:
        file = request.files[img_type]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Prefix filename with student ID and image type to avoid conflicts
            filename = f"{new_student.id}_{img_type}_{filename}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(save_path)
                # Save image record to the database
                image_type = img_type.replace('_image', '')
                new_image = Image(image_filename=filename, image_type=image_type, student_id=new_student.id)
                db.session.add(new_image)
            except Exception as e:
                flash(f'Error saving file {img_type.replace("_", " ").capitalize()}: {e}')
                continue
        else:
            flash(f'Invalid file type for {img_type.replace("_", " ").capitalize()}')
            continue
    
    # Commit all image records
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        flash(f'Error saving images to database: {e}')
        return redirect(request.url)
    
    # After uploading, update the model
    update_model()
    flash('Files successfully uploaded and model updated')
    return redirect(url_for('upload_form'))

def update_model():
    folderPath = app.config['UPLOAD_FOLDER']
    students = Student.query.all()
    imgList = []
    studentIds = []

    for student in students:
        images = Image.query.filter_by(student_id=student.id).all()
        for image in images:
            img_path = os.path.join(folderPath, image.image_filename)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    imgList.append(img)
                    studentIds.append(student.id)
                else:
                    print(f"Failed to load image for {student.name} (ID: {student.id})")
            else:
                print(f"Image file does not exist: {img_path}")

    if not imgList:
        return

    # Encode the images
    encodeListKnown = findEncodings(imgList)
    encodeListKnownWithIds = [encodeListKnown, studentIds]

    # Save the encodings
    try:
        with open("EncodeFile.p", 'wb') as file:
            pickle.dump(encodeListKnownWithIds, file)
    except Exception as e:
        print(f"Error saving encoding file: {e}")


def findEncodings(imagesList):
    encodeList = []

    for img in imagesList:
        # Resize image to reduce processing time and memory usage
        small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

        # Convert to grayscale for contrast adjustment
        gray_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_img = clahe.apply(gray_img)

        # Apply denoising to remove noise and artifacts that could affect recognition
        denoised_img = cv2.fastNlMeansDenoising(equalized_img, None, 30, 7, 21)

        # Convert back to RGB after contrast improvement and denoising
        img_rgb = cv2.cvtColor(denoised_img, cv2.COLOR_GRAY2RGB)

        # Normalize the image to a standard range [0, 1]
        img_rgb = img_rgb / 255.0

        # Detect face locations with a 'cnn' model for higher accuracy (requires GPU)
        face_locations = face_recognition.face_locations(img_rgb, model='cnn')

        # If a face is found, detect facial landmarks as a secondary verification
        if face_locations:
            # Extract facial landmarks to confirm accurate detection
            face_landmarks = face_recognition.face_landmarks(img_rgb, face_locations)
            if face_landmarks:
                # Encode the face using the detected face locations
                encodings = face_recognition.face_encodings(img_rgb, known_face_locations=face_locations)

                if encodings:
                    # Store the first encoding, assuming one face per image
                    encodeList.append(encodings[0])

    return encodeList


@app.route('/attendance')
def attendance():
    # Fetch all attendance records
    records = db.session.query(Attendance, Student).join(Student).order_by(Attendance.starttime.desc()).all()
    return render_template('attendance.html', records=records)

# Mark attendance function
@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    student_id = request.form['student_id']
    try:
        # Check if the student is already logged in
        attendance_record = Attendance.query.filter_by(student_id=student_id, endtime=None).first()
        if attendance_record:
            attendance_record.endtime = datetime.utcnow()
            flash(f'Attendance logged out for student ID {student_id}')
        else:
            new_attendance = Attendance(student_id=student_id)
            db.session.add(new_attendance)
            flash(f'Attendance logged in for student ID {student_id}')
        db.session.commit()
    except Exception as e:
        flash(f'Error marking attendance: {e}')
    return redirect(url_for('attendance'))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5021, debug=True)
