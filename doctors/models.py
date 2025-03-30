import datetime

from django.contrib.postgres.fields import ArrayField
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
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='uploads')
    file_name = models.CharField(max_length=100)
    # delete file and migrate
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(default=now)
    # related_name='uploads' allows access to a patient's scans with patient.uploads.all()
    file_url = models.URLField(default='http://example.com/')
    # prediction = models.CharField(max_length=255, null=True, blank=True)
    prediction = ArrayField(models.CharField(max_length=200), blank=True)
    # stored as [result, result_confidence, other_confidence, tumor_results]

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['patient', 'file_name', 'file'], name='unique_upload_name_per_patient')
        ]

    def __str__(self):
        return f"{self.file.name} for {self.patient.get_full_name()}"


# class Note(models.Model):
#     patient = models.ForeignKey('Patient', on_delete=models.CASCADE, related_name='notes')
#     content = models.TextField()
#     created_at = models.DateTimeField(auto_now_add=True)
#     updated_at = models.DateTimeField(auto_now=True)
#
#     def __str__(self):
#         return self.content[:50]

class Note(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='notes')
    content = models.TextField()
    # created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
