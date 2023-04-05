from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


x_train = ['O filme é ótimo','Não gostei do filme','A trama é intrigante','A atuação é triste','Foi um bom filme']
y_train = ['positivo','negativo','positivo','negativo','positivo']

x_test = ['Atuação triste','Não recomendo o filme','Muito bom']

vectorizer = CountVectorizer()

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

model = MultinomialNB()

model.fit(x_train_vec, y_train)

y_pred = model.predict(x_test_vec )

print(y_pred)