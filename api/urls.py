from django.urls import path
from routers import classifier_router, meme_generator_router, chat_router

# all rest apis here
urlpatterns=[
    path('', classifier_router.getData),
    path('classify/', classifier_router.classify_meme), #  http://127.0.0.1:8000/api/classify
    path('generate/meme/text', meme_generator_router.generate_meme_text),
    path('send/',chat_router.send)
]