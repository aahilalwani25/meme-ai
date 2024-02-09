from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework.decorators import api_view
from nudity_detection.nude_detector import NudeDetector
from django.core.files.storage import FileSystemStorage
from os import path



@api_view(['GET'])
def getData(request):
    return Response({
        "data":'Hello'
    })


@api_view(['POST'])
def classify_meme(request: Request):
    meme = request.FILES.get('meme_photo')
   
    if meme:
        # Define the file storage location within your project
        folder=path.dirname('server/images')
        fs = FileSystemStorage(location=folder)
        
        # Save the uploaded file to the specified location
        filename = fs.save(meme.name, meme)
        
        # Now, 'filename' contains the relative path to the saved file in your project folder.
        response = 'POST API and you have uploaded a {} file, saved as {}'.format(meme.name, filename)
        detector = NudeDetector()
        detections = detector.detect(f'{folder}/{filename}')
        
        return Response((detections))
    else:
        response = 'POST API and no image file was uploaded'

    return Response({
        "message": response
    })
    