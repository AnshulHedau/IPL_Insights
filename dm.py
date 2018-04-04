import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset= pd.read_csv('matches.csv')
dataset=dataset.fillna('no')
teams=pd.unique(dataset[['team1', 'team2']].values.ravel('K'))
dataset=dataset.replace(to_replace=teams, value=[0,1,2,3,4,5,6,7,8,9,10,11,12,13])             

X=dataset.iloc[:,0:18].values
X_new=dataset.iloc[:,[2,4,5,6,7,13]].values
X=np.delete(X,[2,3,4,5,6,7,8,9,13,14,15],axis=1)              
Y=dataset[['winner']]              
X=dataset.iloc[:,10:12].values
Y=Y.replace(to_replace='no',value=16)
Y=Y.values              
Y=Y.reshape(-1,1)       
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X_new[:,0] = labelencoder.fit_transform(X_new[:,0])
labelencoder2=LabelEncoder()
X_new[:,1] = labelencoder2.fit_transform(X_new[:,1])
labelencoder5=LabelEncoder()
X_new[:,2] = labelencoder.fit_transform(X_new[:,2])
labelencoder6=LabelEncoder()
X_new[:,3] = labelencoder2.fit_transform(X_new[:,3])
labelencoder7=LabelEncoder()
X_new[:,4] = labelencoder7.fit_transform(X_new[:,4])
labelencoder8=LabelEncoder()
X_new[:,5] = labelencoder7.fit_transform(X_new[:,5])



'''
onehotencoder = OneHotEncoder(categorical_features = [0,1,2,3,4,5,6,7,8,9,10])
X_new = onehotencoder.fit_transform(X_new).toarray()
'''


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size = 0.1, random_state = 0)


'''
T=X_new[0:508,:]
T2=X_new[508:,:]
X_train=np.concatenate((X_train,T),axis=1)
X_test=np.concatenate((X_test,T2),axis=1)

'''



from sklearn.naive_bayes import GaussianNB as GB
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
ne=RF(n_estimators=90)
ne.fit(X_train,y_train)


y_pred=ne.predict(X_test)
from sklearn.metrics import accuracy_score
acc= accuracy_score(y_test,y_pred)  
final=pd.DataFrame()
final['team1']=X_test[:,1]
final['team2']=X_test[:,2]
#final['venue']=X_test[:,5]
final['winner']=y_pred
ven=labelencoder7.inverse_transform(list(X_test[:,5])) 
ven=list(ven)    
final=final.replace(to_replace=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], value=teams)
final['venue']=ven
