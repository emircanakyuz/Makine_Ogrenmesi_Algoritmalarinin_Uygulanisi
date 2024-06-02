import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc

# Veri setini yükleme
veri = pd.read_excel('Pima Indians Diabetes Dataset.xlsx')
# Veri setimizin genel bilgilerine bakalım..
print(veri.info())
# Veri setimizin temel istatistiklerine göz atalım..
print(veri.describe())
# Sınıf kategorisini inceleylim..
print(veri['Class variable (0 or 1)'].value_counts())

# features olarak belirlediğimizd eğişkende, içeriğinde 0 olması pek de muhtemel olmayan sütunlarımız bulunuyor.
# Mesela, Diyastolik kan basıncının 0 olması da mümkün değildir. Eğer bir kişinin diyastolik kan basıncı 0 ise, bu kan dolaşımının olmadığı anlamına gelir ki bu hayatta kalmak için imkansızdır.
# Bundan dolayı Diastolic blood pressure sütununu bu listeye dahil ettik. Listede bulunan diğer sütunlar da bu şekilde eklenmiştir.
features = ['Plasma glucose concentration a 2 hours in an oral glucose tolerance test',
            'Diastolic blood pressure (mm Hg)',
            'Triceps skinfold thickness (mm)',
            '2-Hour serum insulin (mu U/ml)',
            'Body mass index (weight in kg/(height in m)^2)']
# Eksik verileri sütun medyanı ile değiştirme işlemi..
for feature in features:
    median = veri[feature].median()
    veri[feature] = veri[feature].replace(0, median)

# Aykırı değerleri Çeyrekler Arası Aralık (IQR) yöntemi ile tespit ediyoruz..
q1 = veri.quantile(0.25) # (Birinci Çeyrek): Veri setinin %25'lik dilimindeki değerdir. Yani, veri setindeki değerlerin %25'i bu değerden düşüktür.
q3 = veri.quantile(0.75) # Üçüncü Çeyrek): Veri setinin %75'lik dilimindeki değerdir. Yani, veri setindeki değerlerin %75'i bu değerden düşüktür.
iqr = q3 - q1 # IQR, birinci çeyrek (Q1) ile üçüncü çeyrek (Q3) arasındaki farktır ve veri setinin orta %50'sinin yayılımını gösterir.
# 1.5 katsayısı genellikle istatistikte yaygın olarak kullanıldığından ve çoğu durumda iyi sonuçlar verir, biz de bundan dolayı 1.5 kullandık. Ancak tabii ki bu sayı veri setinize göre değiştirebilecek bir hiper parametredir.
lower_bound = q1 - 1.5 * iqr # Bu değerin altında kalan değerler aykırı kabul edilir.
upper_bound = q3 + 1.5 * iqr # Bu değerin altında kalan değerler aykırı kabul edilir.
outliers = ((veri < lower_bound) | (veri > upper_bound)).any(axis=1) # Veri setindeki her bir sütun için lower_bound ve upper_bound sınırlarının dışında kalan değerleri tespit eder. Yani outliers değişkeni, her satır için en az bir sütunda aykırı değer olup olmadığını gösteren bir boolean dizisidir.
veri = veri[~outliers] # Belirlenen aykırı değerlere ait satırlar veri setinden silinir..

# Şimdi ise dengesiz olan verilerimizi dengeli hale getirmek amacııyla sentetik veri üretelim..
smote = SMOTE(random_state=42) # ilgili sınıfı kullanmak için smote nesnesi oluşturalım.
X = veri.drop('Class variable (0 or 1)', axis=1) # Bağımsız değişkenleri X' e atadık
y = veri['Class variable (0 or 1)'] # Bağımlı değişkenleri y' ye atadık
# Veri arttırımıını gerçekleştiriyoruz. Smote sınıfı sentetik şekilde veri üretimi işlemini K-en yakın komşu algoritmasını uygukayarak gerçekleştirmektedir.
# Rastgele bir k komşusu seçilir. Seçilen örnek ile bu komşu arasındaki her bir özellik farkı hesaplanır. Bu farkın rastgele bir kısmı (0 ile 1 arasında bir sayı ile çarpılarak) seçilen örneğin özelliklerine eklenir.
# Bu işlem, özellik uzayında, var olan örnek ile seçilen komşusu arasında yeni bir nokta oluşturur. Bu yeni nokta, orijinal veri noktalarına benzer özelliklere sahip sentetik bir örnektir.
X, y = smote.fit_resample(X, y)
print(pd.Series(y).value_counts())

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42) # Veri setimizi train ve test olarak ayırıyoruz.

# Min-Max normalizasyonu (Preprocessing işlemlerinden 5. sıradaki...)
normalizasyon = MinMaxScaler()
X_train = normalizasyon.fit_transform(X_train)
X_test = normalizasyon.transform(X_test)
# Model üzerinden çağıracağımız fit fonksiyonu girdi verilerini 2 boyutlu beklerken, çıktı verilerini tek boyutlu beklemektedir. Bundan dolayı ravel() metodunu kullanıyoruz.
y_train = y_train.ravel() 

# İlgili algoritmalarımızı kullanabilmek için sınıflarından nesneler oluşturuyoruz..
model_MLP = MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=1000, random_state=42, activation='relu')
model_KNN= KNeighborsClassifier(n_neighbors=5, metric='euclidean')
model_SVM = SVC(kernel='linear', random_state = 42)
model_NaiveBayes = GaussianNB()

#---------------------------------------------------------------------------------
# Naive Bayes Algoritmasının eğitimi ve performans hesaplamaları..
#---------------------------------------------------------------------------------
model_NaiveBayes.fit(X_train, y_train) # Eğitim işlemini gerçekleştiriyoruz..
y_pred_KNN_NaiveBayes = model_NaiveBayes.predict(X_test) # Eğitilmiş model ile tahmin gerçekleştiriyoruz

# Naive Bayes Algoritmasının confusion matrix hesabı
conf_matrix = confusion_matrix(y_test, y_pred_KNN_NaiveBayes)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.4)
ax = sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='coolwarm', cbar=False)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.xaxis.set_ticklabels(['Negative', 'Positive'])
ax.yaxis.set_ticklabels(['Negative', 'Positive'])
plt.show()

# Naive Bayes Algoritmasının Performans Metrikleri
accuracy = accuracy_score(y_test, y_pred_KNN_NaiveBayes)
precision = precision_score(y_test, y_pred_KNN_NaiveBayes)
recall = recall_score(y_test, y_pred_KNN_NaiveBayes)
f1 = f1_score(y_test, y_pred_KNN_NaiveBayes)
sensitivity = recall
specificity = recall_score(y_test, y_pred_KNN_NaiveBayes, pos_label=0)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall/Sensitivity: {recall}")
print(f"Specificity: {specificity}")
print(f"F1 Score: {f1}")

# Naive Bayes Algoritmasının ROC Eğrisi hesabı..
fpr, tpr, thresholds = roc_curve(y_test, y_pred_KNN_NaiveBayes)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#---------------------------------------------------------------------------------
# KNN Algoritmasının eğitimi ve performans hesaplamaları..
#---------------------------------------------------------------------------------
model_KNN.fit(X_train, y_train)
y_pred_KNN = model_KNN.predict(X_test)

#KNN Algoritmasının confusion matrix hesaBI
conf_matrix = confusion_matrix(y_test, y_pred_KNN)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.4)
ax = sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='coolwarm', cbar=False)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.xaxis.set_ticklabels(['Negative', 'Positive'])
ax.yaxis.set_ticklabels(['Negative', 'Positive'])
plt.show()

# KNN Algoritmasının Performans Metrikleri
accuracy = accuracy_score(y_test, y_pred_KNN)
precision = precision_score(y_test, y_pred_KNN)
recall = recall_score(y_test, y_pred_KNN)
f1 = f1_score(y_test, y_pred_KNN)
sensitivity = recall
specificity = recall_score(y_test, y_pred_KNN, pos_label=0)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall/Sensitivity: {recall}")
print(f"Specificity: {specificity}")
print(f"F1 Score: {f1}")

# KNN Algoritmasının ROC Eğrisi hesabı..
fpr, tpr, thresholds = roc_curve(y_test, y_pred_KNN)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#---------------------------------------------------------------------------------
# MLP Algoritmasının eğitimi ve performans hesaplamaları..
#---------------------------------------------------------------------------------
model_MLP.fit(X_train, y_train) # Eğitim işlemini gerçekleştiriyoruz..
y_pred_MLP = model_MLP.predict(X_test) # Eğitilmiş model ile tahmin gerçekleştiriyoruz

# MLP Algoritmasının confusion matrix hesabı
conf_matrix = confusion_matrix(y_test, y_pred_MLP)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.4)
ax = sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='coolwarm', cbar=False)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.xaxis.set_ticklabels(['Negative', 'Positive'])
ax.yaxis.set_ticklabels(['Negative', 'Positive'])
plt.show()

# MLP Algoritmasının Performans Metrikleri
accuracy = accuracy_score(y_test, y_pred_MLP)
precision = precision_score(y_test, y_pred_MLP)
recall = recall_score(y_test, y_pred_MLP)
f1 = f1_score(y_test, y_pred_MLP)
sensitivity = recall
specificity = recall_score(y_test, y_pred_MLP, pos_label=0)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall/Sensitivity: {recall}")
print(f"Specificity: {specificity}")
print(f"F1 Score: {f1}")

# MLP Algoritmasının ROC Eğrisi hesabı..
fpr, tpr, thresholds = roc_curve(y_test, y_pred_MLP)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#---------------------------------------------------------------------------------
# MLP Algoritmasının eğitimi ve performans hesaplamaları..
#---------------------------------------------------------------------------------
model_SVM.fit(X_train, y_train) # Eğitim işlemini gerçekleştiriyoruz..
y_pred = model_SVM.predict(X_test) # Eğitilmiş model ile tahmin gerçekleştiriyoruz

# SVM algoritmasının confusion matrix hesabı..
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.4)
ax = sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='coolwarm', cbar=False)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.xaxis.set_ticklabels(['Negative', 'Positive'])
ax.yaxis.set_ticklabels(['Negative', 'Positive'])
plt.show()

# SVM Algoritmasının Performans Metrikleri
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
sensitivity = recall
specificity = recall_score(y_test, y_pred, pos_label=0)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall/Sensitivity: {recall}")
print(f"Specificity: {specificity}")
print(f"F1 Score: {f1}")

# SVM Algoritmasının ROC Eğrisi hesabı..
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()