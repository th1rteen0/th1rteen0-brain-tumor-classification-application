from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('new_scan/', views.new_scan, name='new_scan'),
    path('create_patient_file/', views.create_patient_file, name='create_patient_file'),
    path('patient_search/', views.patient_search, name='patient_search'),
    path('patient_file/<int:patient_id>/', views.patient_file, name='patient_file'),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
