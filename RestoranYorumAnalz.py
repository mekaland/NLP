import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Gerekli nltk paketlerini indir
nltk.download("stopwords")
nltk.download("wordnet")

# Lemmatizer oluştur
lema = nltk.WordNetLemmatizer()

# Veriyi yükle
data = pd.read_csv("C:/Users/hp/OneDrive/Masaüstü/Anlaşılır ekonomi/Restaurant_Reviews.tsv", delimiter="\t")

veri = data.copy()
temiz = []

# Yorumları temizle
for i in range(len(veri)):
    duzenle = re.sub('[^a-zA-Z]', ' ', veri["Review"][i])
    duzenle = duzenle.lower()
    duzenle = duzenle.split()
    duzenle = [lema.lemmatize(kelime) for kelime in duzenle if not kelime in set(stopwords.words("english"))]
    duzenle = ' '.join(duzenle)
    temiz.append(duzenle)

# Frekans analizi ve görselleştirme
df = pd.DataFrame(list(zip(veri["Review"], temiz)), columns=["Orjinal yorum", "Temiz yorum"])

frekans = (df["Temiz yorum"]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
frekans.columns = ["Kelimeler", "Frekans"]

filtre = frekans[frekans["Frekans"] > 10]
filtre.plot.bar(x="Kelimeler", y="Frekans")
plt.show()

# Veriyi vektöre çevir
cv = CountVectorizer(max_features=1500)
matrix = cv.fit_transform(temiz).toarray()  # bağımsız değişken
y = veri.iloc[:, 1].values  # bağımlı değişken

# Veriyi ayır
X_train, X_test, y_train, y_test = train_test_split(matrix, y, test_size=0.2, random_state=0)

# Model oluştur ve tahmin yap
model = MultinomialNB()
model.fit(X_train, y_train)
tahmin = model.predict(X_test)

# Doğruluk skoru
skor = accuracy_score(y_test, tahmin)
print("Doğruluk Skoru: ", skor * 100)
