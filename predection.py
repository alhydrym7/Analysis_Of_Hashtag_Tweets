# import pickle
# import nltk
# import re
# import pandas as pd


# cv = pickle.load(open('cv_transforms.pkl', 'rb'))
# nb_classifier = pickle.load(open('arabic-text-classifier.pkl', 'rb'))




# while True:
#     text_to_predict = input("Enter Your Text: ")
#     if text_to_predict == 0 :
#         break
#     arabic_stopwords = set(nltk.corpus.stopwords.words('arabic'))
#     arabic_stemmer = nltk.ISRIStemmer()

#     text = re.sub(pattern=r'[^\u0600-\u06FF\u0750-\u077F\s]', repl='', string=text_to_predict)
#     text = text.lower()
#     words = text.split()
#     words = [word for word in words if word not in arabic_stopwords]
#     words = [arabic_stemmer.stem(word) for word in words]
#     cleaned_text = ' '.join(words)

#     text_vector = cv.transform([cleaned_text]).toarray()


#     predicted_class = nb_classifier.predict(text_vector)

#     class_mapper = {0: 'normal', 1: 'abusive'}
#     predicted_class_text = class_mapper[predicted_class[0]]

#     print(f"النص: {text_to_predict}")
#     print(f"التصنيف المتوقع: {predicted_class_text}")



import pickle
import nltk
import re
import pandas as pd
import numpy as np

# قم بتحميل الـ CountVectorizer ونموذج الـ Naive Bayes من ملفات البيكل الخاصة بهما
cv = pickle.load(open('cv_transforms.pkl', 'rb'))
nb_classifier = pickle.load(open('arabic-text-classifier.pkl', 'rb'))

# قم بتحميل مجموعة الاختبار (تأكد من أنها تحتوي على النصوص التي ترغب في التوقع لها)
test_data = pd.read_csv(r'C:\Users\asus\Desktop\Artificial intelligence\Third Year\Summer\EVC\Dataanlysez\Task-3\data\test.csv')  # استبدل path_to_test_data.csv بمسار ملف البيانات الخاص بك

# تنفيذ نفس العمليات المستخدمة في بناء النموذج لتنظيف النصوص
arabic_stopwords = set(nltk.corpus.stopwords.words('arabic'))
arabic_stemmer = nltk.ISRIStemmer()

corpus = []
for i in range(test_data.shape[0]):
    text = re.sub(pattern=r'[^\u0600-\u06FF\u0750-\u077F\s]', repl='', string=test_data['Tweet'][i])
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in arabic_stopwords]
    words = [arabic_stemmer.stem(word) for word in words]
    cleaned_text = ' '.join(words)
    corpus.append(cleaned_text)

# تحويل النصوص المنظفة إلى مصفوفة باستخدام CountVectorizer
test_vectors = cv.transform(corpus).toarray()

# التنبؤ باستخدام النموذج لمجموعة الاختبار
predicted_probabilities = nb_classifier.predict_proba(test_vectors)

# طباعة نسبة التوقع لكل جملة في مجموعة الاختبار
for i, tweet in enumerate(test_data['Tweet']):
    print(f"الجملة: {tweet}")
    print(f"نسبة التوقع للتصنيف 'normal': {predicted_probabilities[i][0]:.4f}")
    print(f"نسبة التوقع للتصنيف 'abusive': {predicted_probabilities[i][1]:.4f}")
    print("--------------------------------------------")
