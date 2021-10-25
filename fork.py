import pandas as pd 
import numpy as np

from wordcloud import WordCloud # for wordclouds
import re
from sklearn.manifold import TSNE
import sklearn
from matplotlib.colors import ListedColormap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('alldata.csv')
positions = data['position'].value_counts()
top = list(positions.index)[:20]
positions_text = " ".join(data['position'].dropna().to_list())
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(positions_text)
positions_text = " ".join(data['company'].dropna().to_list())
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(positions_text)
description = data['description']

#there are so many job profiles in teh given dataset so lets Categories them into 5; Data Scientist, Machine Learning Engineer, Data Analyst, Data Science Manager and Others

# Creating only 5 datascience roles among all
data2 = data.copy()
data2.dropna(subset=['position'], how='all', inplace = True)
data2['position']=[x.upper() for x in data2['position']]
data2['description']=[x.upper() for x in data2['description']]

data3 = data2.copy()
# Data Scientist keywords
data2.loc[data2.position.str.contains("SCIENTIST"), 'position'] = 'Data Scientist'
data2.loc[data2.position.str.contains('DATA SCIENCE'),'position'] = 'Data Scientist'
# Machine learning engineer keywords
data2.loc[data2.position.str.contains('ENGINEER'),'position']='Machine Learning Engineer'
data2.loc[data2.position.str.contains('PRINCIPAL STATISTICAL PROGRAMMER'),'position']='Machine Learning Engineer'
data2.loc[data2.position.str.contains('PROGRAMMER'),'position']='Machine Learning Engineer'
data2.loc[data2.position.str.contains('DEVELOPER'),'position']='Machine Learning Engineer'
# Data analyst keywords
data2.loc[data2.position.str.contains('ANALYTICS'), 'position'] = 'Data Analyst'
data2.loc[data2.position.str.contains('ANALYST'), 'position'] = 'Data Analyst'
data2.loc[data2.position.str.contains('STATISTICIAN'), 'position'] = 'Data Analyst'
data2.loc[data2.position.str.contains('RESEARCH ASSOCIATE'), 'position'] = 'Data Analyst'
data2.loc[data2.position.str.contains('SUPPLY'), 'position'] = 'Data Analyst'
data2.loc[data2.position.str.contains('RISK'), 'position'] = 'Data Analyst'
data2.loc[data2.position.str.contains('MARKET'), 'position'] = 'Data Analyst'
# Data science manager keywords are the most varying
data2.loc[data2.position.str.contains('MANAGER'),'position']='Data Science Manager'
data2.loc[data2.position.str.contains('MANAGEMENT'),'position']='Data Science Manager'
data2.loc[data2.position.str.contains('CONSULTANT'),'position']='Data Science Manager'
data2.loc[data2.position.str.contains('DIRECTOR'),'position']='Data Science Manager'
data2.loc[data2.position.str.contains('PRESIDENT'),'position']='Data Science Manager'
data2.loc[data2.position.str.contains('EXECUTIVE'),'position']='Data Science Manager'
data2.loc[data2.position.str.contains('HEAD'),'position']='Data Science Manager'
data2.loc[data2.position.str.contains('COORDINATOR'),'position']='Data Science Manager'
data2.loc[data2.position.str.contains('LEADER'),'position']='Data Science Manager'
data2.loc[data2.position.str.contains('OFFICER'),'position']='Data Science Manager'
data2.loc[data2.position.str.contains('CHIEF'),'position']='Data Science Manager'



data2.position=data2[(data2.position == 'Data Scientist') | (data2.position == 'Data Analyst')
                     | (data2.position == 'Machine Learning Engineer') | (data2.position == 'Data Science Manager')]
data2.position=['Others' if x is np.nan else x for x in data2.position]

position=data2.groupby(['position'])['company'].count()
position=position.reset_index(name='company')
position=position.sort_values(['company'],ascending=False)

position3=data3.groupby(['position'])['company'].count()
position3=position3.reset_index(name='company')
position3=position3.sort_values(['company'],ascending=False)

data2['position_raw'] = data3['position']


tfidf=TfidfVectorizer()
y_embed = tfidf.fit_transform(data3['position'])
position_embedded = TSNE(n_components=2).fit_transform(y_embed)
data2['position_tsne_1'] = position_embedded[:, 0]
data2['position_tsne_2'] = position_embedded[:, 1]

print('Here is  the count of each new roles we created :', '\n\n', position)

# Next Part in ML Algorithm is Data Cleansing
X=data2.description
Y=data2.position
X=[re.sub(r"[^a-zA-Z0-9]+", ' ', k) for k in X]
X=[re.sub("[0-9]+",' ',k) for k in X]

#applying stemmer
ps =PorterStemmer()
X=[ps.stem(k) for k in X]

#Note: I have not removed stop words because there are important key words mentioned in job description which are of
# length 2, I feel they have weightage while classifing
label_enc=LabelEncoder()
X=tfidf.fit_transform(X)

description_embedded = TSNE(n_components=2).fit_transform(X)
data2['description_tsne_1'] = description_embedded[:, 0]
data2['description_tsne_2'] = description_embedded[:, 1]


# add a position_raw = '<position> Centroid' data point for each class
# position,company,description,reviews,location,position_raw,position_tsne_1,position_tsne_2,
# description_tsne_1,description_tsne_2
DAp1 = np.mean(data2.loc[data2.position.str.contains('Data Analyst'),'position_tsne_1'])
DAp2 = np.mean(data2.loc[data2.position.str.contains('Data Analyst'),'position_tsne_2'])
DAd1 = np.mean(data2.loc[data2.position.str.contains('Data Analyst'),'description_tsne_1'])
DAd2 = np.mean(data2.loc[data2.position.str.contains('Data Analyst'),'description_tsne_2'])
data2.loc[len(data2.index)] = ['Data Analyst', "", "", 0, "", 'Data Analyst Centroid', DAp1, DAp2, DAd1, DAd2]
DSp1 = np.mean(data2.loc[data2.position.str.contains('Data Scientist'),'position_tsne_1'])
DSp2 = np.mean(data2.loc[data2.position.str.contains('Data Scientist'),'position_tsne_2'])
DSd1 = np.mean(data2.loc[data2.position.str.contains('Data Scientist'),'description_tsne_1'])
DSd2 = np.mean(data2.loc[data2.position.str.contains('Data Scientist'),'description_tsne_2'])
data2.loc[len(data2.index)] = ['Data Scientist', "", "", 0, "", 'Data Scientist Centroid', DSp1, DSp2, DSd1, DSd2]
DMp1 = np.mean(data2.loc[data2.position.str.contains('Data Science Manager'),'position_tsne_1'])
DMp2 = np.mean(data2.loc[data2.position.str.contains('Data Science Manager'),'position_tsne_2'])
DMd1 = np.mean(data2.loc[data2.position.str.contains('Data Science Manager'),'description_tsne_1'])
DMd2 = np.mean(data2.loc[data2.position.str.contains('Data Science Manager'),'description_tsne_2'])
data2.loc[len(data2.index)] = ['Data Science Manager', "", "", 0, "", 'Data Science Manager Centroid', DMp1, DMp2, DMd1, DMd2]
MLEp1 = np.mean(data2.loc[data2.position.str.contains('Machine Learning Engineer'),'position_tsne_1'])
MLEp2 = np.mean(data2.loc[data2.position.str.contains('Machine Learning Engineer'),'position_tsne_2'])
MLEd1 = np.mean(data2.loc[data2.position.str.contains('Machine Learning Engineer'),'description_tsne_1'])
MLEd2 = np.mean(data2.loc[data2.position.str.contains('Machine Learning Engineer'),'description_tsne_2'])
data2.loc[len(data2.index)] = ['Machine Learning Engineer', "", "", 0, "", 'Machine Learning Engineer Centroid', MLEp1, MLEp2, MLEd1, MLEd2]
Op1 = np.mean(data2.loc[data2.position.str.contains('Others'),'position_tsne_1'])
Op2 = np.mean(data2.loc[data2.position.str.contains('Others'),'position_tsne_2'])
Od1 = np.mean(data2.loc[data2.position.str.contains('Others'),'description_tsne_1'])
Od2 = np.mean(data2.loc[data2.position.str.contains('Others'),'description_tsne_2'])
data2.loc[len(data2.index)] = ['Others', "", "", 0, "", 'Others Centroid', Op1, Op2, Od1, Od2]

data2.to_csv('alljobs.csv')

Y=label_enc.fit_transform(Y)

#SGD classification
n_members = 30
models = []
for i in range(n_members):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.3)
    # define and fit model
    sgd = SGDClassifier()
    sgd.fit(x_train, y_train)
    # store model in memory as ensemble member
    models.append(sgd)

# TEST
from scipy import stats
yhats = [model.predict(x_test) for model in models]
yhats = np.array(yhats)
# calculate average prediction over bagging ensemble
sgd_y = stats.mode(yhats, axis=0)[0][0]
print('Accuracy of SGD :', accuracy_score(y_test,sgd_y))
print('Confusion Matrix of SGD : ', '\n\n', confusion_matrix(y_test,sgd_y))
for row in confusion_matrix(y_test,sgd_y):
    print(np.max(row)/np.sum(row))

np.savetxt("cm.csv", confusion_matrix(y_test,sgd_y), delimiter=",")

# # where are errors occuring?
# X = tfidf.inverse_transform(x_test)
# for i in range(len(sgd_y)):
#     if y_test[i] != sgd_y[i] and y_test[i]==1.:
#         print(X[i])


