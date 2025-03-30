import mimetypes
import os
import tensorflow as tf
from tensorflow import keras
from django.shortcuts import get_object_or_404, render, redirect
import boto3
from .models import Patient, Upload, Note
from .forms import PatientUploadForm, UploadForm, NoteForm
import keras.models
from keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import tempfile
from mysite.settings import AWS_STORAGE_BUCKET_NAME, AWS_S3_REGION_NAME
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import ast
from django.db.models import Q, Value
from django.db.models.functions import Concat
import mimetypes
from django.http import FileResponse, HttpResponseForbidden
from django.conf import settings
from botocore.exceptions import BotoCoreError, ClientError
import io
from django.urls import reverse
from django.template.loader import render_to_string
from django.contrib import messages


binary_model = load_model('models/binary_model.keras')
tumor_model = load_model('models/multi_class_braintumormodel.keras')


def dashboard(request):
    return render(request, 'dashboard.html')


def new_scan(request):
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
        result, result_confidence, other_confidence, tumor_results = get_prediction_for_scan(scanned_file)

        content = {
            'scanned_file_url': scanned_file_url,
            'filename': filename,
            'patients': patients,
            'result': result,
            'result_confidence': result_confidence,
            'other_confidence': other_confidence,
            'tumor_results': tumor_results,
        }

        return render(request, 'new_scan.html', content)

    if request.method == 'POST' and request.POST.get('assign_scan'):
        assign_scan = request.POST.get('assign_scan')
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

            # URL to the uploaded file
            scanned_file_url = f'https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_path}'

            # Get existing prediction details (stored in POST data)
            result = request.POST.get('result')
            result_confidence = request.POST.get('result_confidence')
            other_confidence = request.POST.get('other_confidence')
            tumor_results = request.POST.getlist('tumor_results')

            # Create prediction array
            prediction = [result, result_confidence, other_confidence, tumor_results]

            # Save the scan metadata to the database
            Upload.objects.create(patient=patient, file_name=filename, file_url=scanned_file_url, prediction=prediction)

            return redirect('new_scan')

        except Exception as e:
            print(f'Error uploading to S3: {e}')

    return render(request, 'new_scan.html', {'patients': patients})


def get_prediction_for_scan(scan_file):
    result = None
    result_confidence = None
    other_confidence = None
    tumor_results = None

    if scan_file:
        result, result_confidence, other_confidence, tumor_results = process_scan(scan_file)

    return result, result_confidence, other_confidence, tumor_results


def process_scan(scan_file):
    # Convert uploaded file to a PIL image
    image = Image.open(scan_file)
    # Run tumor detection and classification
    result, result_confidence, other_confidence, tumor_results = detect_tumor(image)
    return result, result_confidence, other_confidence, tumor_results


def detect_tumor(image):
    tumor_confidence, no_tumor_confidence = predict_tumor(image)

    if tumor_confidence >= 0.5:
        # If tumor is detected, run through the second model for tumor classification
        tumor_type, tumor_type_confidence = classify_tumor_type(image)

        # Create a list to hold the tumor type results and confidence levels
        tumor_results = [{
            'type': tumor_type,
            'confidence': f"{tumor_type_confidence * 100:.2f}%",
        }]

        # Display the other tumor types with their confidence levels
        tumor_types = ["Glioma", "Meningioma", "Pituitary"]
        for i, t in enumerate(tumor_types):
            if t != tumor_type:
                other_tumor_confidence = tumor_model.predict(preprocess_image(image))[0][i] * 100
                tumor_results.append({
                    'type': t,
                    'confidence': f"{other_tumor_confidence:.2f}%",
                })

        result = "Brain Tumor Detected"
        result_confidence = f"{tumor_confidence * 100:.2f}%"
        other_confidence = f"{no_tumor_confidence * 100:.2f}%"

    else:
        # If no tumor is detected, display the result and the possible tumor types
        result = "No Brain Tumor Detected"
        result_confidence = f"{no_tumor_confidence * 100:.2f}%"
        other_confidence = f"{tumor_confidence * 100:.2f}%"

        # Display the confidence for possible tumor types (in case it wasn't detected as a tumor)
        tumor_types = ["Glioma", "Meningioma", "Pituitary"]
        tumor_predictions = tumor_model.predict(preprocess_image(image))[0]

        tumor_results = [{
            'type': t,
            'confidence': f"{tumor_predictions[i] * 100:.2f}%",
        } for i, t in enumerate(tumor_types)]

    return result, result_confidence, other_confidence, tumor_results


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
    patients = Patient.objects.annotate(
        full_name=Concat('first_name', Value(' '), 'last_name')  # Create a full_name field
    )

    # apply search filter
    search_query = request.GET.get('q', '')
    if search_query:
        patients = patients.filter(
            Q(full_name__icontains=search_query) |
            Q(first_name__icontains=search_query) |
            Q(last_name__icontains=search_query)
        )

    return render(request, 'patient_search.html', {'patients': patients, 'search_query': search_query})


def patient_file(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id)
    prediction = None
    file_url = None
    formatted_tumor_results = None

    try:
        # Attempt to get the most recent scan
        latest_scan = Upload.objects.filter(patient=patient).latest('uploaded_at')
        formatted_tumor_results = format_prediction(latest_scan)
    except Upload.DoesNotExist:
        latest_scan = None

    # if latest_scan:
    #     s3 = boto3.client('s3', region_name=AWS_S3_REGION_NAME)
    #     bucket_name = AWS_STORAGE_BUCKET_NAME
    #
    #     # Determine the S3 file key from the file_obj's path
    #     file_key = f'patient_{patient_id}/{latest_scan.file_name}'
    #
    #     file_url = s3.generate_presigned_url(
    #         'get_object',
    #         Params={
    #             'Bucket': bucket_name,
    #             'Key': file_key,
    #             # makes sure the browser is able to handle the display the correct media type
    #             'ResponseContentType': 'image/jpeg',
    #             'ResponseContentDisposition': 'inline',
    #         },
    #         ExpiresIn=3600  # URL expires in 1 hour
    #

    # Handle file upload and prediction
    # if request.method == 'POST' and request.FILES.get('scan_file'):
    #     scan_file = request.FILES['scan_file']
    #     fs = FileSystemStorage()
    #     filename = fs.save(scan_file.name, scan_file)
    #
    #     # Get prediction for the new scan
    #     result, result_confidence, other_confidence, tumor_results = get_prediction_for_scan(scan_file)
    #
    #     # Upload to S3
    #     s3 = boto3.client('s3', region_name=AWS_S3_REGION_NAME)
    #     file_path = f'patient_{patient_id}/{filename}'
    #
    #     try:
    #         temp_file_path = fs.path(filename)
    #         s3.upload_file(temp_file_path, AWS_STORAGE_BUCKET_NAME, file_path)
    #         scanned_file_url = f'https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_path}'
    #
    #         prediction = [result, result_confidence, other_confidence, tumor_results]
    #
    #         latest_scan = Upload.objects.create(patient=patient, file_name=filename, file_url=scanned_file_url,
    #                                             prediction=prediction)
    #
    #     except Exception as e:
    #         print(f"Error uploading to S3: {e}")
    #         file_url = None  # Ensure file_url is None if upload fails

    # Retrieve All Notes
    notes = patient.notes.all()  # Retrieve all prompts related to this upload

    # Handle note form submission
    if request.method == 'POST' and 'add_note' in request.POST:
        note_form = NoteForm(request.POST)
        if note_form.is_valid():
            new_note = note_form.save(commit=False)
            new_note.patient = patient
            # new_note.created_by = request.user
            new_note.save()

            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                # Re-fetch notes to include the new one
                notes = patient.notes.all()

                # Render the partial template
                notes_html = render_to_string('partials/notes_partial.html', {
                    'notes': notes,
                    'note_form': NoteForm(),
                    'patient': patient,
                }, request=request)

                return JsonResponse({'html': notes_html})

            return redirect('patient_file', patient_id=patient.id)
    else:
        note_form = NoteForm()

    context = {
        'patient': patient,
        'latest_scan': latest_scan,
        # 'file_url': file_url if file_url else None,
        'formatted_tumor_results': formatted_tumor_results,
        'note_form': note_form,
        'notes': notes,
    }

    return render(request, 'patient_file.html', context)


def delete_note(request, note_id):
    if request.method == 'POST':
        note = get_object_or_404(Note, id=note_id)

        patient_id = note.patient.id
        note.delete()

        return redirect('patient_file', patient_id=patient_id)


def edit_note(request, note_id):
    note = get_object_or_404(Note, id=note_id)

    if request.method == 'POST':
        content = request.POST.get('content')
        if content:
            note.content = content
            note.save()
            messages.success(request, 'Note updated successfully.')
        else:
            messages.error(request, 'Note content cannot be empty.')
        return redirect('patient_file', patient_id=note.patient.id)


def edit_patient(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id)

    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        phone_number = request.POST.get('phone_number')
        address = request.POST.get('address')

        # Update patient details
        patient.first_name = first_name
        patient.last_name = last_name
        patient.email = email
        patient.phone_number = phone_number
        patient.address = address
        patient.save()

        # Show success message and redirect
        messages.success(request, 'Patient information updated successfully.')
        return redirect('patient_file', patient_id=patient_id)
    else:
        messages.error(request, 'Invalid request.')
        return redirect('patient_file', patient_id=patient_id)


def secure_patient_image(request, patient_id, file_name):
    """ Fetches and streams the file from S3 to the user without exposing the S3 URL. """

    file_key = f'patient_{patient_id}/{file_name}'
    s3 = boto3.client('s3', region_name=settings.AWS_S3_REGION_NAME)
    bucket_name = settings.AWS_STORAGE_BUCKET_NAME

    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(file_key)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default binary type

    try:
        # Fetch file from S3
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        file_stream = io.BytesIO(response['Body'].read())  # Read file into memory

        # Serve the file through Django (hiding S3 URL)
        return FileResponse(file_stream, content_type=mime_type)

    except (BotoCoreError, ClientError) as e:
        print(f"Error retrieving file from S3: {e}")
        return HttpResponseForbidden("Unable to access the requested file.")


def format_prediction(latest_scan):
    tumor_results = latest_scan.prediction[3]
    tumor_results = ast.literal_eval(tumor_results)

    final_list = []

    for parsed in tumor_results:
        new_parsed = ast.literal_eval(parsed)
        for each in new_parsed:
            result_list = []
            for values in each.values():
                result_list.append(values)
            results = result_list[0] + ": " + result_list[1]
            final_list.append(results)

    return final_list


def all_files(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id)
    uploads = patient.uploads.all()

    upload_with_secure_urls = []

    for upload in uploads:
        try:
            # Generate a Django URL for secure image access
            secure_url = reverse('secure_patient_image', args=[patient_id, upload.file_name])

            # Ensure the prediction is a valid dictionary
            prediction_data = upload.prediction
            if isinstance(prediction_data, str):
                import ast
                prediction_data = ast.literal_eval(prediction_data)

            formatted_tumor_results = format_prediction(upload)

            upload_with_secure_urls.append({
                'upload': upload,
                'secure_url': secure_url,  # Use Django URL instead of S3 presigned URL
                'formatted_tumor_results': formatted_tumor_results,
                'prediction': prediction_data
            })

        except Exception as e:
            print(f"Error processing file {upload.file_name}: {e}")
            upload_with_secure_urls.append({
                'upload': upload,
                'secure_url': None,
                'formatted_tumor_results': None,
                'prediction': None
            })
    return render(request, 'all_files.html', {'upload_with_secure_urls': upload_with_secure_urls, 'patient': patient})


def delete_patient(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id)

    s3 = boto3.client('s3', region_name=AWS_S3_REGION_NAME)
    bucket_name = AWS_STORAGE_BUCKET_NAME

    file_key = f'patient_{patient_id}'

    try:
        patient_to_delete = s3.list_objects_v2(Bucket=bucket_name, Prefix=file_key)
        if 'Contents' in patient_to_delete:
            delete_keys = [{'Key': obj['Key']} for obj in patient_to_delete['Contents']]

            s3.delete_objects(Bucket=bucket_name, Delete={'Objects': delete_keys})

        patient.delete()
    except Exception as e:
        print(f"Error deleting patient file from S3: {e}")

    return redirect('patient_search')