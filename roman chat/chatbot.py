import pickle
import random
import json
import numpy as np
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from nltk import word_tokenize
from pydantic import BaseModel
import sys
sys.path.append(r'roman chat')

lemmatizer= WordNetLemmatizer()

intents= json.loads(open(r'roman chat/roman.json').read())
words= pickle.load(open(r'roman chat/words.pkl','rb'))
classes= pickle.load(open(r'roman chat/classes.pkl','rb'))
model= load_model(r'roman chat/chatbot_model.h5')

def clean_up_sentences(sentence):
    sentence_words= word_tokenize(sentence)
    sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]

    return sentence_words

def bag_of_words(sentence):
    sentence_words= clean_up_sentences(sentence=sentence)
    bag= [0] * len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word==w:
                bag[i]=1

    return np.array(bag)

def predict_class(sentence):
    bow= bag_of_words(sentence=sentence)
    res= model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD= 0.25
    result= [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    result.sort(key=lambda x:x[1], reverse=True)
    return_list=[]
    for r in result:
        return_list.append(
            {
                'intent':classes[r[0]],
                'probability':str(r[1])
            })
        
    return return_list

def get_response(intents_list):
    tag= intents_list[0]['intent']
    list_of_intents= intents['intents']
    result= None
    for i in list_of_intents:
        if i['tag']==tag:
            result= random.choice(i['responses'])
            break
    return result

class Message(BaseModel):
    message:str

