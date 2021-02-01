from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.home, name='home'),
    path('gallery', views.gall, name='gallery'),
    path('save/', views.save, name='save'),
    path('generar/', views.home, name='generar'),
    path("cargar/", views.image_upload_view, name='cargar'),
]

if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL, document_root=settings.MEDIA_ROOT
    ) + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
