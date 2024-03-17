from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework.decorators import api_view
from nudity_detection.nude_detector import NudeDetector
from django.core.files.storage import FileSystemStorage
from os import path
import sys

sys.path.append(r'roman chat')
sys.path.append(r'meme classification')

from meme_classification_main import get_prediction


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
        file_path= f'{folder}/{filename}'
        detections = detector.detect(file_path)
        if(detector.is_nude(detections)):
            return Response({
                "status":'NEGATIVE',
                "reason": 'The picture consist of sensitive part that might be offended by the people',
                "calculation":{
                    
                }
            })
        else:
            #now predicting on the basis of memes
            prediction= get_prediction(file_path)    #   Predicting the sentiment of the image.
            prediction=prediction.upper()   #   Capitalising the first letter of the string.
            prediction=prediction.replace("_", " ") #   Replacing the underscore with a white-space.
            return Response({
                "status":prediction,
                "calculation":{

                }
            })
    else:
        response = 'POST API and no image file was uploaded'

    return Response({
        "message": response
    })
