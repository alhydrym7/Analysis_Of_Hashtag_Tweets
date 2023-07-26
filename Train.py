# تحميل المكتبات اللازمة
import pandas as pd
import pickle
import nltk
import re

# قم بتحميل ملف "train_arabic.csv" الذي يحتوي على البيانات باللغة العربية
# واستبدل المسار التالي بالمسار الصحيح للملف
df = pd.read_csv(r'C:\Users\asus\Desktop\Artificial intelligence\Third Year\Summer\EVC\Dataanlysez\Task-3\data\train.csv')

# Importing essential libraries
import pandas as pd
import pickle
import nltk
import re

# Downloading the Arabic stopwords list
nltk.download('stopwords')




# Mapping the classes to values
class_mapper = {'normal': 0, 'abusive': 1}
df['Class'] = df['Class'].map(class_mapper)

# Removing the 'id' column if it exists
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Cleaning the Arabic text
corpus = []
arabic_stopwords = set(nltk.corpus.stopwords.words('arabic'))
arabic_stemmer = nltk.ISRIStemmer()

for i in range(df.shape[0]):
    # Cleaning special characters from the Arabic text
    text = re.sub(pattern=r'[^\u0600-\u06FF\u0750-\u077F\s]', repl='', string=df['Tweet'][i])

    # Converting the entire text into lowercase
    text = text.lower()

    # Tokenizing the text by words
    words = text.split()

    # Removing the Arabic stopwords
    words = [word for word in words if word not in arabic_stopwords]

    # Stemming the words
    words = [arabic_stemmer.stem(word) for word in words]

    # Joining the stemmed words
    text = ' '.join(words)

    # Creating a corpus
    corpus.append(text)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=10000, ngram_range=(1, 2))
X = cv.fit_transform(corpus).toarray()
y = df['Class'].values

# Creating a pickle file for the CountVectorizer
pickle.dump(cv, open('cv_transforms.pkl', 'wb'))

# Model Building
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Drop rows with NaN values in the target variable
X_train = X_train[~pd.isnull(y_train)]
y_train = y_train[~pd.isnull(y_train)]

# Fitting Naive Bayes to the Training set
nb_classifier = MultinomialNB(alpha=0.1)
nb_classifier.fit(X_train, y_train)

# Creating a pickle file for the Multinomial Naive Bayes model
filename = 'arabic-text-classifier.pkl'
pickle.dump(nb_classifier, open(filename, 'wb'))

from sklearn.metrics import accuracy_score

# الخطوات السابقة للتدريب وبناء النموذج

# Predicting on the training set
y_train_pred = nb_classifier.predict(X_train)

# Calculating the accuracy on the training set
train_accuracy = accuracy_score(y_train, y_train_pred)

print("دقة التدريب الكامل:", train_accuracy)



# import pandas as pd
# from nltk.tokenize import word_tokenize
# data = pd.read_csv(r'data\train.csv')



