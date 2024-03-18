from django.urls import path
from web_scraper.views import get_funny_content
from routers import classifier_router, meme_generator_router, chat_router

# all rest apis here
urlpatterns=[
    path('', classifier_router.getData),
    path('classify/', classifier_router.classify_meme), #  http://127.0.0.1:8000/api/classify
    path('get_funny_content/<str:query>/', get_funny_content, name='get_funny_content'),
    path('generate/meme/text', meme_generator_router.generate_meme_text),
    path('send/',chat_router.send)
]