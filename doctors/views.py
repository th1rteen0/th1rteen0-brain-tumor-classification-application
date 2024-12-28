import os
from django.shortcuts import get_object_or_404, render, redirect
import boto3
from .models import Patient, Upload
from .forms import PatientUploadForm, UploadForm
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import tempfile
from mysite.settings import AWS_STORAGE_BUCKET_NAME, AWS_S3_REGION_NAME
from django.core.files.storage import FileSystemStorage


binary_model = load_model('models/binary_model.keras')
tumor_model = load_model('models/multi_class_braintumormodel.keras')


def dashboard(request):
    return render(request, 'dashboard.html')


def new_scan(request):
    prediction = None
    patients = Patient.objects.all()

    if request.method == 'POST' and request.FILES.get('scanned_file'):
        scanned_file = request.FILES['scanned_file']
        fs = FileSystemStorage()

        # Clean up old files in the storage directory
        storage_location = fs.location
        for file_name in os.listdir(storage_location):
            file_path = os.path.join(storage_location, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

        filename = fs.save(scanned_file.name, scanned_file)
        scanned_file_url = fs.url(filename)  # URL to access the uploaded file

        # Get the prediction for the scan file using the helper function
        prediction = get_prediction_for_scan(scanned_file)

        return render(request, 'new_scan.html', {'prediction': prediction, 'scanned_file_url': scanned_file_url,
                                                 'filename': filename, 'patients': patients})

    if request.method == 'POST' and request.POST.get('assign_scan'):
        # Get the patient ID and the filename
        patient_id = request.POST.get('patient_id')
        filename = request.POST.get('filename')
        patient = Patient.objects.get(id=patient_id)  # Get the patient from the database

        # Upload the file to S3
        s3 = boto3.client('s3', region_name=AWS_S3_REGION_NAME)
        file_path = f'patient_{patient_id}/{filename}'  # Create a folder structure based on patient ID

        try:
            # Upload to S3
            fs = FileSystemStorage()
            temp_file_path = fs.path(filename)  # Get the temporary file path
            s3.upload_file(temp_file_path, AWS_STORAGE_BUCKET_NAME, file_path)

            # # URL to the uploaded file
            # scanned_file_url = f'https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_path}'

            # Save the scan metadata to the database
            Upload.objects.create(patient=patient, file_name=filename, prediction=prediction)

            return redirect('new_scan')

        except Exception as e:
            print(f'Error uploading to S3: {e}')

    return render(request, 'new_scan.html', {'patients': patients})


def get_prediction_for_scan(scan_file):
    prediction = None

    if scan_file:
        prediction = process_scan(scan_file)

    return prediction


def process_scan(scan_file):
    # Convert uploaded file to a PIL image
    image = Image.open(scan_file)
    # Run tumor detection and classification
    result = detect_tumor(image)
    return result


def detect_tumor(image):
    tumor_confidence, no_tumor_confidence = predict_tumor(image)

    if tumor_confidence >= 0.5:
        # If tumor is detected, run through the second model for tumor classification
        tumor_type, tumor_type_confidence = classify_tumor_type(image)

        # Format the results for tumor detection and type
        result = "Results: Brain Tumor Detected"
        result += f"\nBrain Tumor Detected with {tumor_confidence * 100:.2f}% confidence"
        result += f"\nNo Brain Tumor Detected with {no_tumor_confidence * 100:.2f}% confidence"
        result += f"\nTumor Type: {tumor_type} with {tumor_type_confidence * 100:.2f}% confidence"

        # Display the tumor types with confidence levels
        tumor_types = ["Glioma", "Meningioma", "Pituitary"]
        for i, t in enumerate(tumor_types):
            if t != tumor_type:
                result += f"\n\tOther Tumor Type: {t} with {tumor_model.predict(preprocess_image(image))[0][i] * 100:.2f}% confidence"

    else:
        # If no tumor is detected, display the result and the possible tumor types
        result = "Results: No Brain Tumor Detected"
        result += f"\nNo Brain Tumor Detected with {no_tumor_confidence * 100:.2f}% confidence"
        result += f"\nBrain Tumor Detected with {tumor_confidence * 100:.2f}% confidence"

        # Display the confidence for possible tumor types (in case it wasn't detected as a tumor)
        tumor_types = ["Glioma", "Meningioma", "Pituitary"]
        tumor_predictions = tumor_model.predict(preprocess_image(image))[0]
        for i, t in enumerate(tumor_types):
            result += f"\n\tPossible Tumor Type: {t} with {tumor_predictions[i] * 100:.2f}% confidence"

    return result


# Function to predict if there is a brain tumor
def predict_tumor(image):
    # Preprocess the image
    img = preprocess_image(image)

    # Get the model's prediction (output will be between 0 and 1)
    prediction = binary_model.predict(img)

    # The model's confidence for the tumor class (class 1)
    tumor_confidence = prediction[0][0]  # Since it's a single output, get the first element

    # The confidence for the "no tumor" class (class 0)
    no_tumor_confidence = 1 - tumor_confidence

    return tumor_confidence, no_tumor_confidence


# Function to preprocess an image
def preprocess_image(image):
    # If the image is a PIL image
    if isinstance(image, Image.Image):
        # Save the uploaded image to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            image.save(temp_file, format="JPEG")  # Save image as JPEG
            temp_file_path = temp_file.name

        # Load the image using OpenCV
        img = cv2.imread(temp_file_path)

        # Resize the image to the input shape the model expects
        img_resized = cv2.resize(img, (256, 256))

        # Normalize the image to [0, 1] by dividing by 255
        img_normalized = img_resized / 255.0

        # Add an extra dimension to the image to match the input format (batch_size, height, width, channels)
        img_expanded = np.expand_dims(img_normalized, axis=0)

        # Delete the temporary file after use
        os.remove(temp_file_path)

        return img_expanded
    else:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            for chunk in image.chunks():
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        # Load the image using OpenCV
        img = cv2.imread(temp_file_path)

        # Resize the image to the input shape the model expects
        img_resized = cv2.resize(img, (256, 256))

        # Normalize the image to [0, 1] by dividing by 255
        img_normalized = img_resized / 255.0

        # Add an extra dimension to the image to match the input format (batch_size, height, width, channels)
        img_expanded = np.expand_dims(img_normalized, axis=0)

        # Delete the temporary file after use
        os.remove(temp_file_path)

        return img_expanded


# Define classify_tumor_type to be a model that classifies tumor types
def classify_tumor_type(image):
    # Load and preprocess the image for the second model
    img = preprocess_image(image)

    # Get the tumor type prediction
    prediction = tumor_model.predict(img)
    tumor_types = ["Glioma", "Meningioma", "Pituitary"]  # Example classes

    # Get the index of the highest confidence (softmax output)
    tumor_type_index = np.argmax(prediction)

    # Confidence for the predicted tumor type
    tumor_type_confidence = prediction[0][tumor_type_index]

    # Tumor type corresponding to the class index
    tumor_type = tumor_types[tumor_type_index]

    # Return the tumor type and the confidence level
    return tumor_type, tumor_type_confidence


def create_patient_file(request):
    if request.method == 'POST':
        form = PatientUploadForm(request.POST)
        if form.is_valid():
            patient = form.save(commit=False)
            patient.save()
            # return to dashboard for now to see if a user has been created properly
            return redirect('patient_search')
    else:
        form = PatientUploadForm()

    return render(request, 'create_patient_file.html', {'form': form})


def patient_search(request):
    patients = Patient.objects.all()
    return render(request, 'patient_search.html', {'patients': patients})


def patient_file(request, patient_id):
    prediction = None
    patient = get_object_or_404(Patient, id=patient_id)

    if request.method == 'POST' and request.FILES.get('scan_file'):
        scan_file = request.FILES['scan_file']

        prediction = get_prediction_for_scan(scan_file)

    return render(request, 'patient_file.html', {'patient': patient, 'prediction': prediction})

