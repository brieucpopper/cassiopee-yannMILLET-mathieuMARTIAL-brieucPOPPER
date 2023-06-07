import pandas as pd

#merge finalDbwithSetntimentMerged and topic.csv
finalDb=pd.read_csv('finalDbWithSentimentMerged.csv')
topic=pd.read_csv('topic.csv',sep=';')

#merge by link
finalDb=finalDb.merge(topic, on='link', how='left')

#save
#DROP THE COLUMNS THAT ARE NOT USEFUL
finalDb=finalDb.drop(['Unnamed: 0_x','Unnamed: 0_y','Unnamed: 0.1'],axis=1)
finalDb.to_csv('finalfinalDB.csv',index=False)




import pandas as pd

#open finalfinalDB


def cleanAndPerformOneHot():
    finalDb=pd.read_csv('finalfinalDB.csv')


    #below some code to quickly fix a minor merge error on one line that happened

    # #fix merge error
    # #set sentiment to P
    # finalDb.loc[finalDb['sentiment'].isnull(),'sentiment']='P'
    # #set the score to 5561094880104065
    # finalDb.loc[finalDb['sentiment'].isnull(),'score']=0.5561094880104065

    #check for misisng data
    print("be careful, below is missing data:")
    print(finalDb.isnull().sum())

    import sklearn

    #convert sentiment to 0/1 if negative/positive
    finalDb['sentiment']=finalDb['sentiment'].apply(lambda x: 1 if x=='P' else 0)


    #this adds in the number of subscribers in mid 2022, from socialblade
    def mapSubs(n):
        if n=='lefigarofr':
            return 594450/1613000
        if n=='lemondefr':
            return 1613000/1613000
        if n=='lesechos':
            return 99987/1613000
        if n=='lhumanitefr':
            return 40000/1613000
        if n=='liberationfr':
            return 445770/1613000
        if n=='publicfr':
            return 183248/1613000
        if n=='valeurs_actuelles':
            return 42331/1613000
        

    #for these newspaper, map the amount of subs
    finalDb['subs']=finalDb['newspaper'].apply(mapSubs)

    #one hot encode newspaper
    print(finalDb['newspaper'].unique())
    finalDb=pd.get_dummies(finalDb,columns=['newspaper'])


    finalDb['likesPerSubs']=finalDb['nb_likes']/finalDb['subs']
    finalDb['commentsPerSubs']=finalDb['nb_post_comments']/finalDb['subs']

    #group hours by 4 (otherwise we have too much dimensions and less relevant analysis)
    finalDb['hourGroup']=finalDb['hour'].apply(lambda x: 'nuit' if x<4 else 'tresTot' if x<8 else 'matin' if x<12 else 'aprem' if x<16 else 'finAprem' if x<20 else 'soirTard')
    finalDb=pd.get_dummies(finalDb,columns=['hourGroup'])

    #one hot encode day of week
    finalDb=pd.get_dummies(finalDb,columns=['dow'])


    #create new topic column (the topics were manually inspected for this, can be modified)
    def convertTopics(t):
        if t==3:
            return 'guerreUkraine'
        if t==4:
            return 'electionsPresidentielles'
        if t==5:
            return 'coupeDuMonde'
        if t==6:
            return 'reformeRetraite'
        if t==7:
            return 'electionsParlementaires'
        else:
            return 'autre'

    finalDb['relevantTopic']=finalDb['topic'].apply(convertTopics)
    finalDb=pd.get_dummies(finalDb,columns=['relevantTopic'])

    #group the months by season
    def returnSeason(m):
        if m=='January' or m=='February' or m=='December':
            return 'hiver'
        if m=='March' or m=='April' or m=='May':
            return 'printemps'
        if m=='June' or m=='July' or m=='August':
            return 'ete'
        if m=='September' or m=='October' or m=='November':
            return 'automne'
    finalDb['season']=finalDb['month'].apply(returnSeason)
    finalDb=pd.get_dummies(finalDb,columns=['season'])



    #print the columns
    #print(finalDb.keys())

    #print each newspaper

    #drop publicfr because it isn't political (if it was still there from before)
    finalDb=finalDb.drop(columns=['newspaper_publicfr'])

    independantVars=['link','sentiment','hourGroup_aprem','hourGroup_finAprem','hourGroup_matin',
                    'hourGroup_nuit','hourGroup_soirTard','hourGroup_tresTot',
                    
                    'dow_Friday','dow_Monday','dow_Saturday','dow_Sunday','dow_Thursday'
                    ,'dow_Tuesday','dow_Wednesday','season_automne','season_ete',
                    'season_hiver','season_printemps','timeSinceLastPost','nbCarac',
                    'nbHashtags','nbMentions','nbEmojis',
                    'relevantTopic_autre','relevantTopic_coupeDuMonde',
                    'relevantTopic_electionsParlementaires','relevantTopic_electionsPresidentielles',
                    'relevantTopic_guerreUkraine','relevantTopic_reformeRetraite','newspaper_lefigarofr','newspaper_lesechos','newspaper_lhumanitefr','newspaper_liberationfr','newspaper_valeurs_actuelles']

    
    

    #save only link,nb_likes, nb_post_comments in a separate dataframe
    toPredict=finalDb[['link','nb_likes','nb_post_comments','likesPerSubs','commentsPerSubs','subs']]

    predictWith=finalDb[independantVars]


    #export these two dataframes
    toPredict.to_csv('toPredict.csv',index=False)
    predictWith.to_csv('predictWith.csv',index=False)

cleanAndPerformOneHot()
