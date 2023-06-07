import pandas as pd

import matplotlib.pyplot as plt
import numpy as np


data=pd.read_stata('/home/cassiopee/yann/Insta_posts_Mars2023.dta')
def statsOnDatabase(data):
    #gives some basic information about the database which is a pandas df
    print("The shape of the database is: ",data.shape)
    print("The columns of the database are: ",data.columns)

    #print out randomly a sample of 5 rows with each column printed separately
    print("A sample of 5 rows of the database is: ")

    print(data.head(5))

    #for a specific sample go more in details
    sample=data.head(1)
    keys=sample.keys()

    for key in keys:
        print(key,": ",type(sample[key].values[0]),':',sample[key].values[0])


#statsOnDatabase(data)
"""The shape of the database is:  (26188, 30)
The columns of the database are:  Index(['link', 'newspaper', 'posted_at', 'scraped_at', 'content', 'nb_likes',
       'nb_views', 'nb_post_comments', 'nb_faces', 'tot_newlike_day',
       'tot_newcomment_day', 'nb_news_newlike_day', 'nb_news_newcomment_day',
       'tot_newlike_post', 'tot_newcomment_post', 'post_hour', 'post_day',
       'post_week', 'scrap_hour', 'scrap_day', 'sum_post_comments_like',
       'sum_post_comments_child', 'collected_like', 'over_20000',
       'nb_post_day', 'nb_like_day', 'nb_coll_like_day',
       'sum_post_comments_like_day', 'sum_post_comments_child_day', 'id_day'],
      dtype='object')
A sample of 5 rows of the database is: 
              link       newspaper             posted_at  ... sum_post_comments_like_day sum_post_comments_child_day  id_day
1187   CZzZkUjhHa7           voici  2022-02-10T16:15:58Z  ...                     6655.0                       413.0    34.0
10430  Cfoyp8Zo1JK           voici  2022-07-05T16:30:57Z  ...                     7199.0                       875.0    44.0
4746   CcJUYPzoVyb         lequipe  2022-04-09T21:35:45Z  ...                     1900.0                       327.0    22.0
7862   CeInYhbImhD  madamefigarofr  2022-05-29T08:05:51Z  ...                    11347.0                       591.0    34.0
13813  CiKuz88AH0i      pointdevue  2022-09-06T13:54:46Z  ...                    22923.0                      1179.0    63.0

[5 rows x 30 columns]
link :  <class 'str'> : CZ-2hWoq0My
newspaper :  <class 'str'> : lequipe
posted_at :  <class 'str'> : 2022-02-15T02:59:39Z
scraped_at :  <class 'str'> : 2022-03-04T11:34:20
content :  <class 'str'> : Déception pour Tess Ledeux qui se classe 7e du ski slopestyle de ces jeux Olympiques.

📸 @franckfaugere / L'Equipe

#Beijing2022
nb_likes :  <class 'numpy.int32'> : 10008
nb_views :  <class 'numpy.int32'> : 0
nb_post_comments :  <class 'numpy.int16'> : 8
nb_faces :  <class 'numpy.float32'> : 0.0
tot_newlike_day :  <class 'numpy.float32'> : 90910.0
tot_newcomment_day :  <class 'numpy.float32'> : 1547.0
nb_news_newlike_day :  <class 'numpy.float32'> : 76102.0
nb_news_newcomment_day :  <class 'numpy.float32'> : 789.0
tot_newlike_post :  <class 'numpy.float32'> : 1609.0
tot_newcomment_post :  <class 'numpy.float32'> : 7.0
post_hour :  <class 'int'> : 1960513241088
post_day :  <class 'numpy.datetime64'> : 2022-02-15T00:00:00.000000000
post_week :  <class 'numpy.datetime64'> : 2022-02-12T00:00:00.000000000
scrap_hour :  <class 'int'> : 1962012835840
scrap_day :  <class 'numpy.datetime64'> : 2022-03-04T00:00:00.000000000
sum_post_comments_like :  <class 'numpy.float32'> : 185.0
sum_post_comments_child :  <class 'numpy.float32'> : 11.0
collected_like :  <class 'numpy.float32'> : 10008.0
over_20000 :  <class 'numpy.float32'> : 0.0
nb_post_day :  <class 'numpy.float32'> : 75.0
nb_like_day :  <class 'numpy.float32'> : 344966.0
nb_coll_like_day :  <class 'numpy.float32'> : 333501.0
sum_post_comments_like_day :  <class 'numpy.float32'> : 10389.0
sum_post_comments_child_day :  <class 'numpy.float32'> : 1321.0
id_day :  <class 'numpy.float32'> : 27.0

['link', 'newspaper', 'posted_at', 'scraped_at', 'content', 'nb_likes',
       'nb_views', 'nb_post_comments', 'nb_faces', 'tot_newlike_day',
       'tot_newcomment_day', 'nb_news_newlike_day', 'nb_news_newcomment_day',
       'tot_newlike_post', 'tot_newcomment_post', 'post_hour', 'post_day',
       'post_week', 'scrap_hour', 'scrap_day', 'sum_post_comments_like',
       'sum_post_comments_child', 'collected_like', 'over_20000',
       'nb_post_day', 'nb_like_day', 'nb_coll_like_day',
       'sum_post_comments_like_day', 'sum_post_comments_child_day', 'id_day'],"""

finalDb=data[['link','newspaper','posted_at','nb_likes','nb_post_comments','sum_post_comments_like','sum_post_comments_child','content']]
#finalDb is the database we will use for the analysis, meaning that we dropped some columns that we aren't going to use


def takeOffQuotes(text):
    return text.replace('"',"")
finalDb['posted_at']=finalDb['posted_at'].apply(takeOffQuotes)
# this takes off some quotes that were a problem for getting the date right





#get all the newspaper
newspaperList=finalDb['newspaper'].unique()
#print("The list of newspaper is: ",newspaperList)
"""['lequipe' 'liberationfr' 'lesechos' 'onzemondial' 'ellefr'
 'Psychologies_' 'marieclairefr' 'parismatch' 'madamefigarofr' 'voici'
 'femme_actuelle' 'santeplusmag' 'pointdevue' 'galafr' 'lefigarofr'
 'valeurs_actuelles' 'lhumanitefr' 'midi.olympique' 'publicfr' 'lemondefr'
 'sofoot' 'autoplusmag' 'enduromag_officiel' 'purepeople']"""
 # a list of all the newspapers we have in the database

politicalNewspapers=['lesechos','liberationfr','lemondefr','lefigarofr','lhumanitefr','valeurs_actuelles','publicfr']
# for further study, we will only take the political newspapers
finalDb=finalDb[finalDb['newspaper'].isin(politicalNewspapers)]



#create 3 new columns, hour of day, day of week, and month


from datetime import datetime

#the function below deals with getting the hour, day of week and month from the string, which can be tricky sometimes
def parse_string(string):
    string = string.strip().strip('"')  # Remove leading and trailing spaces, and the quotation mark
    dt = datetime.strptime(string, "%Y-%m-%dT%H:%M:%SZ")

    hour = dt.hour
    day_of_week = dt.strftime("%A")
    month = dt.strftime("%B")
    toSort=dt.timestamp()

    return hour, day_of_week, month, toSort


finalDb['hour']=finalDb['posted_at'].apply(lambda x: parse_string(x)[0])
finalDb['dow']=finalDb['posted_at'].apply(lambda x: parse_string(x)[1])
finalDb['month']=finalDb['posted_at'].apply(lambda x: parse_string(x)[2])
finalDb['toSort']=finalDb['posted_at'].apply(lambda x: parse_string(x)[3])




#sort by newspaper then by date
# we want to sort like this then reindex to calculate the time since last post
finalDb=finalDb.sort_values(by=['newspaper','toSort'])
#reindex from 0 to len(finalDb)
finalDb=finalDb.reset_index(drop=True)



#GET THE indexes of first appearenace of each newspaper in the database (again to calculate the time since last post)
firstIndexes=[0]
orderedListOfNewspaper=[finalDb.iloc[0]['newspaper']]

currentNewspaper=finalDb.iloc[0]['newspaper']
for i in range(len(finalDb)):
    if(finalDb.iloc[i]['newspaper']!=currentNewspaper):
        currentNewspaper=finalDb.iloc[i]['newspaper']
        orderedListOfNewspaper.append(currentNewspaper)
        firstIndexes.append(i)

print(firstIndexes)
print(orderedListOfNewspaper)

#create column of zeros
finalDb['timeSinceLastPost']=0

#calculate time since last post
for i in range(1,len(finalDb)):
    finalDb.iloc[i,finalDb.columns.get_loc('timeSinceLastPost')]=finalDb.iloc[i]['toSort']-finalDb.iloc[i-1]['toSort']

#delete rows in firstIndexes because they don't have a time since last post
#print the lines at every index in firstIndexes
for i in firstIndexes:
    print(finalDb.iloc[i])
finalDb=finalDb.drop(finalDb.index[firstIndexes])


#adding a new row with the length of the post
finalDb['nbCarac']=finalDb['content'].apply(lambda x: len(x))

#adding a new row with the number of hashtags in the post
def countHashtagsInString(string):
    return string.count('#')

finalDb['nbHashtags']=finalDb['content'].apply(lambda x: countHashtagsInString(x))
finalDb['nbMentions']=finalDb['content'].apply(lambda x: x.count('@'))


#dealing with counting emojis
import emojis
def extract_emojis(s):
  return(len(emojis.get(s)))

finalDb['nbEmojis']=finalDb['content'].apply(lambda x: extract_emojis(x))


statsOnDatabase(finalDb)



finalDb.to_csv('finalDb.csv',index=False)
#use this to save the database to a csv file for further analysis