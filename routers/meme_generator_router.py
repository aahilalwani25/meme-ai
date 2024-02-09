from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework.decorators import api_view



@api_view(['POST'])
def generate_meme_text(request:Request):
    #get a file as a request
    meme_image= request.FILES.get('meme_image')
    
    if not meme_image : 
        return Response({"error":"No image provided"},status=400)
    

    return Response()
