from django.urls import path
from . import views

urlpatterns = [
    #path('admin/', admin.site.urls),
    path('', view=views.getData),
    path('/classify', view=views.classify)
]