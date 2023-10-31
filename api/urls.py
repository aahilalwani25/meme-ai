from django.urls import path
from Controllers import meme_classifier_controller

# all rest apis here
urlpatterns=[
    path('', meme_classifier_controller.getData),
    path('classify/', meme_classifier_controller.classify_meme) #  http://127.0.0.1:8000/api/classify
]