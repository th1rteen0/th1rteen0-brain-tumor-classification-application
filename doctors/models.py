import datetime

from django.db import models
from django.utils.timezone import now
from django.contrib.auth.models import User


class Patient(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    phone_number = models.CharField(max_length=100)
    address = models.CharField(max_length=100)
    # insurance_information

    def __str__(self):
        return f"{self.first_name} {self.last_name}"

    def get_full_name(self):
        return f"{self.first_name} {self.last_name}"


class Upload(models.Model):
    file_name = models.CharField(max_length=100)
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(default=now)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='uploads')
    # related_name='images' allows you to access a patient's images with patient.uploads.all()

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['patient', 'file_name', 'file'], name='unique_upload_name_per_patient')
        ]

    def __str__(self):
        return f"{self.file.name} for {self.patient.get_full_name()}"
