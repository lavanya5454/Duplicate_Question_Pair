import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix

nltk.download("stopwords")
nltk.download("wordnet")



nb_model=MultinomialNB()
cv=CountVectorizer(max_features=2500)
tf=TfidfVectorizer()
le=LabelEncoder()


ps=PorterStemmer()
wl=WordNetLemmatizer()


# Assuming tab-separated values
data = pd.read_csv("/Users/apple/Desktop/spam_ham/SMSSpamCollection", sep='\t',names=["label","message"])
cleaned_corpus=[]
for i in range(0,5572):
    review=re.sub('[^a-zA-Z$]',' ',data['message'][i])
    review=review.lower()
    review=review.split()
    review=[wl.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review=" ".join(review)
    cleaned_corpus.append(review)
x=cv.fit_transform(cleaned_corpus).toarray()  
y=le.fit_transform(data['label'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
nb_model.fit(x_train,y_train)
y_pred=nb_model.predict(x_test)
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
