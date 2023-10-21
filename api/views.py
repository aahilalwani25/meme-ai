from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework.decorators import api_view
import json

@api_view(['GET'])
def getData(request):
    return Response({
        "Hello":"World"
    })

@api_view(['POST'])
def classify(request: Request):
    data = json.loads(request.data)
    return Response({
        'status':200,
        "classification": data['text']
    })

