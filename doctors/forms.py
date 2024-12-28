from django import forms
from .models import Patient, Upload
from django.contrib.auth.models import User


class PatientUploadForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ['first_name', 'last_name', 'email', 'phone_number', 'address']


class UploadForm(forms.ModelForm):
    class Meta:
        model = Upload
        fields = ['patient', 'file_name', 'file']

    file = forms.FileField(required=True)
