from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/analisis/", views.api_analisis, name="api_analisis"),
    path("api/jelaskan/", views.api_jelaskan, name="api_jelaskan"),
]