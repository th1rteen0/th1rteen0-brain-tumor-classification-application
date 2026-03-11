import mimetypes
import os
import tensorflow as tf
from tensorflow import keras
from django.shortcuts import get_object_or_404, render, redirect
import boto3
from .models import Patient, Upload, Note
from .forms import PatientUploadForm, UploadForm, NoteForm
from keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import tempfile
from mysite.settings import AWS_STORAGE_BUCKET_NAME, AWS_S3_REGION_NAME
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, FileResponse, HttpResponseForbidden
import json
import ast
from django.db.models import Q, Value
from django.db.models.functions import Concat
from django.conf import settings
from botocore.exceptions import BotoCoreError, ClientError
import io
from django.urls import reverse
from django.template.loader import render_to_string
from django.contrib import messages
from django.utils import timezone
from functools import lru_cache
import base64
import imutils


# =========================
# MODEL LOADING
# =========================

@lru_cache(maxsize=1)
def get_binary_model():
    return load_model('models/binary_model.keras')


@lru_cache(maxsize=1)
def get_tumor_model():
    return load_model('models/multi_model.keras')


# =========================
# DASHBOARD
# =========================

def dashboard(request):
    patients = Patient.objects.all()
    uploads = Upload.objects.all().order_by('-uploaded_at')[:3]
    return render(request, 'dashboard.html', {'patients': patients, 'uploads': uploads})


# =========================
# SCAN UPLOAD & PREDICTION
# =========================

def new_scan(request):
    patients = Patient.objects.all()

    if request.method == 'POST' and request.FILES.get('scanned_file'):
        scanned_file = request.FILES['scanned_file']
        filename = scanned_file.name

        file_bytes = scanned_file.read()

        # Upload current scan to S3
        s3 = boto3.client('s3', region_name=AWS_S3_REGION_NAME)
        s3.put_object(
            Bucket=AWS_STORAGE_BUCKET_NAME,
            Key='current/scan.jpg',
            Body=file_bytes,
            ContentType='image/jpeg',
        )

        # Generate presigned URL for display
        scanned_file_url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': AWS_STORAGE_BUCKET_NAME, 'Key': 'current/scan.jpg'},
            ExpiresIn=3600,
        )

        image = Image.open(io.BytesIO(file_bytes))
        result, result_confidence, other_confidence, tumor_results, gradcam_b64 = get_prediction_for_scan(image)

        content = {
            'scanned_file_url': scanned_file_url,
            'filename': filename,
            'patients': patients,
            'result': result,
            'result_confidence': result_confidence,
            'other_confidence': other_confidence,
            'tumor_results': tumor_results,
            'gradcam_b64': gradcam_b64,
        }
        return render(request, 'new_scan.html', content)

    if request.method == 'POST' and request.POST.get('assign_scan'):
        patient_id = request.POST.get('patient_id')
        filename = request.POST.get('filename')
        patient = Patient.objects.get(id=patient_id)

        s3 = boto3.client('s3', region_name=AWS_S3_REGION_NAME)
        dest_key = f'patient_{patient_id}/{filename}'

        try:
            s3.copy_object(
                Bucket=AWS_STORAGE_BUCKET_NAME,
                CopySource={'Bucket': AWS_STORAGE_BUCKET_NAME, 'Key': 'current/scan.jpg'},
                Key=dest_key,
            )

            scanned_file_url = f'https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{dest_key}'

            result = request.POST.get('result')
            result_confidence = request.POST.get('result_confidence')
            other_confidence = request.POST.get('other_confidence')
            tumor_results = request.POST.getlist('tumor_results')
            gradcam_b64 = request.POST.get('gradcam_b64')

            prediction = [result, result_confidence, other_confidence, tumor_results]

            Upload.objects.create(patient=patient, file_name=filename, file_url=scanned_file_url, prediction=prediction, gradcam_b64=gradcam_b64)

            return redirect('new_scan')

        except Exception as e:
            print(f'Error assigning scan to patient: {e}')

    return render(request, 'new_scan.html', {'patients': patients})


def get_prediction_for_scan(scan_file):
    try:
        return process_scan(scan_file)
    except Exception as e:
        print(f'Error processing scan: {e}')
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


def process_scan(scan_file):
    if isinstance(scan_file, Image.Image):
        image = scan_file
    else:
        image = Image.open(scan_file)

    return detect_tumor(image)


# =========================
# TUMOR DETECTION PIPELINE
# =========================

def detect_tumor(image):
    model = get_tumor_model()

    tumor_confidence, no_tumor_confidence = predict_tumor(image)

    if tumor_confidence >= 0.5:
        tumor_results, gradcam_b64 = classify_tumor_type(image)
        tumor_results = sorted(tumor_results, key=lambda x: float(x["confidence"].replace("%", "")), reverse=True)
        max_prob = 0
        if tumor_results:
            max_prob = max([float(t["confidence"].replace("%", "")) for t in tumor_results])
        result = "Brain Tumor Detected"
        result_confidence = f"{max_prob:.2f}%"
        other_confidence = f"{no_tumor_confidence * 100:.2f}%"
    else:
        tumor_results = []
        tumor_results = sorted(tumor_results, key=lambda x: float(x["confidence"].replace("%", "")), reverse=True)
        gradcam_b64 = None
        result = "No Brain Tumor Detected"
        result_confidence = f"{no_tumor_confidence * 100:.2f}%"
        other_confidence = f"{tumor_confidence * 100:.2f}%"

    return result, result_confidence, other_confidence, tumor_results, gradcam_b64


def predict_tumor(image):
    """
    Binary model preprocessing:
    - No cropping
    - Resize to 256x256
    - Normalize to 0-1
    Matches training: image_dataset_from_directory with image_size=(256,256) + /255.0
    """
    image = image.convert("RGB")
    img = np.array(image)
    img = cv2.resize(img, (256, 256))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    model = get_binary_model()
    prediction = model.predict(img)
    tumor_confidence = prediction[0][0]
    no_tumor_confidence = 1 - tumor_confidence
    return tumor_confidence, no_tumor_confidence


def classify_tumor_type(image):
    """
    Multi-class model preprocessing:
    - Convert PIL RGB → BGR for OpenCV
    - Crop brain region using contour detection
    - Resize to 240x240
    - Convert BGR → RGB
    - Normalize to 0-1
    Matches training: crop_image() + cv2.resize(240,240) + ImageDataGenerator /255.0
    """
    try:
        tumor_model = get_tumor_model()

        # PIL is RGB, OpenCV expects BGR
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Crop to brain region
        try:
            img_cropped = crop_image(img_cv)
        except Exception as e:
            print(f"Cropping failed, using original image: {e}")
            img_cropped = img_cv

        # Resize to training size
        img_resized = cv2.resize(img_cropped, (240, 240))

        # Convert back to RGB for model
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Normalize
        img_array = img_rgb.astype("float32")
        img_array = np.expand_dims(img_array, axis=0)

        prediction = tumor_model.predict(img_array)[0]
        print("MULTI MODEL RAW PREDICTION:", prediction)

        classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
        tumor_results = []
        for i, prob in enumerate(prediction):
            if classes[i] != "No Tumor":
                tumor_results.append({"type": classes[i], "confidence": f"{prob*100:.2f}%"})

        gradcam_b64 = generate_gradcam_b64(tumor_model, img_array)

        tumor_results = sorted(tumor_results, key=lambda x: float(x["confidence"].replace("%", "")), reverse=True)
        return tumor_results, gradcam_b64

    except Exception as e:
        print(f"classify_tumor_type failed: {e}")
        import traceback
        traceback.print_exc()
        return [], None


# =========================
# UTILS
# =========================

def crop_image(image):
    """
    Crops the brain region from an MRI scan using contour detection.
    Expects BGR image (OpenCV format). Returns BGR image.
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_thresh = cv2.threshold(img_blur, 45, 255, cv2.THRESH_BINARY)[1]
    img_thresh = cv2.erode(img_thresh, None, iterations=2)
    img_thresh = cv2.dilate(img_thresh, None, iterations=2)

    contours = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)

    if not contours:
        return image

    c = max(contours, key=cv2.contourArea)

    extLeft   = tuple(c[c[:, :, 0].argmin()])[0]
    extRight  = tuple(c[c[:, :, 0].argmax()])[0]
    extTop    = tuple(c[c[:, :, 1].argmin()])[0]
    extBottom = tuple(c[c[:, :, 1].argmax()])[0]

    new_img = image[extTop[1]:extBottom[1], extLeft[0]:extRight[0]]

    if new_img.size == 0:
        return image

    return new_img


def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model")


def generate_gradcam_b64(model, img_array, layer_name=None):
    if layer_name is None:
        layer_name = get_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if isinstance(predictions, list):
            predictions = predictions[0]
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap.numpy()

    h, w = img_array.shape[1], img_array.shape[2]
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = np.uint8(255 * img_array[0])
    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    _, buffer = cv2.imencode(".jpg", superimposed)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()


def format_prediction(upload):
    tumor_results = upload.prediction[3]
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

    
# =========================
# PATIENT AND FILE MANAGEMENT
# =========================

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
    formatted_tumor_results = None

    try:
        # Attempt to get the most recent scan
        latest_scan = Upload.objects.filter(patient=patient).latest('uploaded_at')
        formatted_tumor_results = format_prediction(latest_scan)
    except Upload.DoesNotExist:
        latest_scan = None

    # Retrieve All Notes
    notes = Note.objects.filter(patient_id=patient_id).order_by('-created_at')

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
        'formatted_tumor_results': formatted_tumor_results,
        'note_form': note_form,
        'notes': notes,
    }

    return render(request, 'patient_file.html', context)


def delete_patient(request, patient_id):
    # Fetch the patient
    patient = get_object_or_404(Patient, id=patient_id)
    uploads = patient.uploads.all()

    # Initialize S3 client
    s3 = boto3.client('s3', region_name=settings.AWS_S3_REGION_NAME)
    bucket_name = settings.AWS_STORAGE_BUCKET_NAME

    # Delete associated files from AWS S3
    for upload in uploads:
        file_key = f'patient_{patient_id}/{upload.file_name}'
        s3.delete_object(Bucket=bucket_name, Key=file_key)

        upload.delete()  # Delete upload record from the database

    # Delete the patient record
    patient.delete()

    return redirect('patient_search')


def delete_file(request, patient_id, upload_id):
    # Fetch the patient and uploaded file
    patient = get_object_or_404(Patient, id=patient_id)
    upload = get_object_or_404(Upload, id=upload_id)

    # Initialize S3 client
    s3 = boto3.client('s3', region_name=settings.AWS_S3_REGION_NAME)
    bucket_name = settings.AWS_STORAGE_BUCKET_NAME

    # Delete the file from AWS S3
    file_key = f'patient_{patient_id}/{upload.file_name}'
    s3.delete_object(Bucket=bucket_name, Key=file_key)

    upload.delete()  # Delete upload record from the database

    return redirect('all_files', patient_id=patient_id)


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
            note.created_at = timezone.now()
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


def all_files(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id)
    uploads = patient.uploads.all()

    upload_with_secure_urls = []

    if uploads.exists():
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
    else:
        return render(request, 'all_files.html', {'patient': patient})

    return render(request, 'all_files.html', {'upload_with_secure_urls': upload_with_secure_urls, 'patient': patient, 'upload': upload})
