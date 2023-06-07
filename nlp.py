from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine")
model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")

nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

progress=0

import pandas as pd

db=pd.read_csv('topic.csv',sep=';')





def predictAndShowProgress(x):
    x=str(x)
    #limit to the first 1800 characters
    x=x[:1800]
    print(len(x))
    print(x)

    global progress
    progress+=1
    print(progress)
    try:
        return nlp(x)
    except:
        #if there's a problem due to length, cut the last 100 characters than retrry recursively
        progress-=1
        print("RETRYING FOR THE LAST 100 CHARACTERS")
        
    
        return predictAndShowProgress(x[:-100])


import time
import numpy as np
#print db keys
arrayOfCodes=np.array(db['link'])
arrayToPredict=np.array(db['content'])

print(arrayToPredict[0]+"\n\n\n")
predictedArray=np.full((len(arrayToPredict),), '', dtype=str)
predictedArrayScores=np.zeros(len(arrayToPredict))



for i in range(0,len(arrayToPredict)):

    prediction=predictAndShowProgress(arrayToPredict[i])

    predictedArray[i]=prediction[0]['label']
    predictedArrayScores[i]=prediction[0]['score']
    if(i%100==0):
        #save a csv with arrayOfCodes and predictedArray every 100 iterations in case the program has a problem
        #group both in a dataframe
        df = pd.DataFrame({'link': arrayOfCodes, 'sentiment': predictedArray, 'score': predictedArrayScores})
        #save the dataframe
        df.to_csv(r'D:\Downloads\Mathematical Morphology\finalDbWithSentiment.csv')


df = pd.DataFrame({'link': arrayOfCodes, 'sentiment': predictedArray, 'score': predictedArrayScores})
#save the dataframe
df.to_csv('finalDbWithSentiment.csv')

