from django import forms
from .models import Patient, Upload, Note
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


class NoteForm(forms.ModelForm):
    class Meta:
        model = Note
        fields = ['content']
        widgets = {
            'content': forms.Textarea(attrs={
                'placeholder': 'Write your note here...',
                # 'rows': 3,   # Adjust the number of rows as needed
                # 'cols': 50,  # Adjust the number of columns as needed
                'style': 'width: 100%; max-width: 100%;',  # Ensures it’s responsive
            }),
        }
