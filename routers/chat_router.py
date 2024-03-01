from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework.decorators import api_view
from nudity_detection.nude_detector import NudeDetector
from django.core.files.storage import FileSystemStorage
from os import path
import sys

sys.path.append(r'roman chat')

from chatbot import Message, predict_class, get_response

@api_view(['POST'])
def send(message:Message):
    # print(message.message)
    ints= predict_class(message.message)
    res= get_response(intents_list=ints)
    return {
        "name":"AI",
        "message":res
    }