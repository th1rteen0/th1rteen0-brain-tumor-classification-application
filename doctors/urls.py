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
    path('patient-file/<int:patient_id>/notes/', views.manage_notes, name='manage_notes'),
    path('patient-file/<int:patient_id>/all-files', views.all_files, name='all_files'),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
