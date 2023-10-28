from django.urls import path
from . import views

# all rest apis here
urlpatterns=[
    path('', views.getData),
    path('classify/', views.classify_meme)
]