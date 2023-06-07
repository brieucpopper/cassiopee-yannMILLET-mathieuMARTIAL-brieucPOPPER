

import pandas as pd
import numpy as np


#toTest has the variables used for prediciton
def preProcessSingle(toTest):

    
    df1=pd.read_csv('original-predictWith.csv')
    df2=pd.read_csv('original-toPredict.csv')
    #merge the two dataframes
    df1=df1.merge(df2,on='link',how='left')
    goal=['link','nb_likes','nb_post_comments','likesPerSubs','commentsPerSubs','subs']
    #delete one hotencoded columns
    df1=df1.drop(['hourGroup_tresTot','dow_Saturday','season_ete','relevantTopic_autre'],axis=1)
    
    


    orderedistByInfluence=[]
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression


    #split the data
    
    #print a sample of X

    


    #THE COLUMN TO PREDCIT IS HERE : here comments per subscribers
    y=df1['commentsPerSubs']
    





    droppedVars=['relevantTopic_autre', 'season_hiver', 'hourGroup_aprem', 'dow_Wednesday', 'nbCarac', 'nbHashtags', 'relevantTopic_electionsParlementaires', 'season_ete', 'timeSinceLastPost', 'dow_Saturday', 'relevantTopic_electionsPresidentielles', 'relevantTopic_guerreUkraine', 'relevantTopic_reformeRetraite', 'season_automne', 'relevantTopic_coupeDuMonde', 'hourGroup_tresTot', 'nbMentions', 'dow_Monday', 'hourGroup_matin', 'dow_Friday', 'dow_Thursday', 'hourGroup_finAprem', 'season_printemps', 'hourGroup_nuit', 'dow_Tuesday', 'nbEmojis', 'hourGroup_soirTard', 'sentiment', 'dow_Sunday']
    addedVars=toTest
    #here we drop every variable except the ones given as argument
    X=df1.drop(goal,axis=1)
    #take addedVars out of droppedVars
    droppedVars=[x for x in droppedVars if x not in addedVars]
    X=X.drop(droppedVars,axis=1)
        

    print("\n\ndoing regression with dropped vars, meaning we don't use these variables : ,"+str(droppedVars))
    

    print()
    print()

    print("variables for predicition are : ",X.columns)

    

    #divide the data into train and test, test size is the fraction of the data used for testing
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)

    
    




    import statsmodels.api as sm

    #create the model
    X_train=sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train)
    

    results=model.fit()
    print(results.summary())
    #the linear regression is done here
    #do a regression on the test data
    X_test=sm.add_constant(X_test)
    y_pred=results.predict(X_test)
    #print the r2 score
    print("r2 score with the statsmodel regression  : ",sklearn.metrics.r2_score(y_test,y_pred))



    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LinearRegression, Lasso
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    from sklearn.metrics import r2_score

    # Define the parameter grids for each model
    linear_params = {}
    lasso_params = {'alpha': [0.1, 1.0, 10.0]}  # Example alpha values
    #rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}  # Example parameter values
    rf_params = {'n_estimators': [300], 'max_depth': [10]}  # Example parameter values
    xgb_params = {'n_estimators': [1000], 'max_depth': [9], 'learning_rate': [0.01, 0.1, 0.3]}

    # Initialize the models
    linear_reg = LinearRegression()
    lasso_reg = Lasso()
    rf_reg = RandomForestRegressor()
    xgb_reg = xgb.XGBRegressor()

    # Define the grid search for each model
    linear_grid = GridSearchCV(linear_reg, linear_params, scoring='r2')
    lasso_grid = GridSearchCV(lasso_reg, lasso_params, scoring='r2')
    rf_grid = GridSearchCV(rf_reg, rf_params, scoring='r2')
    xgb_grid = GridSearchCV(xgb_reg, xgb_params, scoring='r2')

    # Fit the models and perform grid search
    linear_grid.fit(X_train, y_train)
    lasso_grid.fit(X_train, y_train)
    rf_grid.fit(X_train, y_train) 
    xgb_grid.fit(X_train, y_train)

    # Print the best R^2 scores for each model
    print("Linear Regression - R^2 score: ", r2_score(y_test, linear_grid.predict(X_test)))
    print("Lasso Regression - R^2 score: ", r2_score(y_test, lasso_grid.predict(X_test)))
    print("Random Forest - R^2 score: ", r2_score(y_test, rf_grid.predict(X_test)))
    print("XGBoost - R^2 score:", r2_score(y_test, xgb_grid.predict(X_test)))







preProcessSingle(['hourGroup_aprem', 'dow_Wednesday','season_hiver','relevantTopic_autre','nbCarac','nbHashtags','relevantTopic_electionsParlementaires','season_ete', 'timeSinceLastPost','dow_Saturday','hourGroup_tresTot'])
#preProcessSingle(['newspaper_lefigarofr','newspaper_lesechos','newspaper_lhumanitefr','newspaper_liberationfr','newspaper_valeurs_actuelles','relevantTopic_autre', 'season_hiver', 'hourGroup_aprem', 'dow_Wednesday', 'nbCarac', 'nbHashtags', 'relevantTopic_electionsParlementaires', 'season_ete', 'timeSinceLastPost', 'dow_Saturday', 'relevantTopic_electionsPresidentielles', 'relevantTopic_guerreUkraine', 'relevantTopic_reformeRetraite', 'season_automne', 'relevantTopic_coupeDuMonde', 'hourGroup_tresTot', 'nbMentions', 'dow_Monday', 'hourGroup_matin', 'dow_Friday', 'dow_Thursday', 'hourGroup_finAprem', 'season_printemps', 'hourGroup_nuit', 'dow_Tuesday', 'nbEmojis', 'hourGroup_soirTard', 'sentiment', 'dow_Sunday'])


