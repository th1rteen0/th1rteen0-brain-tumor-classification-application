from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('new-scan/', views.new_scan, name='new_scan'),
    path('create-patient-file/', views.create_patient_file, name='create_patient_file'),
    path('patient-search/', views.patient_search, name='patient_search'),
    path('patient-file/<int:patient_id>/', views.patient_file, name='patient_file'),
    # path('patient-file/<int:patient_id>/notes/', views.manage_notes, name='manage_notes'),
    path('patient-file/<int:patient_id>/all-files', views.all_files, name='all_files'),
    path('patient-file/<int:patient_id>/delete/', views.delete_patient, name='delete_patient'),
    path('patient/<int:patient_id>/secure-image/<str:file_name>/', views.secure_patient_image, name='secure_patient_image'),
    path('delete_note/<int:note_id>/', views.delete_note, name='delete_note'),
    path('edit_note/<int:note_id>/', views.edit_note, name='edit_note'),
    path('edit_patient/<int:patient_id>/', views.edit_patient, name='edit_patient'),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
