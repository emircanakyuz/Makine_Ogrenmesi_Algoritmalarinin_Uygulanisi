# Makine Ögrenmesi Algoritmalarının (KNN, Naive Bayes, VSM, MLP) Uygulanışı
Merhebalar. Bu çalışmamızda belli başlı makine öğrenmesi algoritmalarının preprocessing işlemleri ile birlikte uygulanışını inceleyeceğiz ve sonuçları analiz edeceğiz. Hazırsanız başlayalım..
Algoritmaların detaylarına geçmeden önce "Preprocessing" işlemlerinden bahsetmek istiyorum. Proprocessing, bir veri setini uygulanışa hazır bir hale getirmektir. Daha detaylı açıklayacak olursak; veri temizleme, veri dönüşümü, veri çoğaltma ve bunlara benzer işlemlerin tümüne preprocessing yani veri ön işleme denmektedir. Veri ön işleme hem makine öğrenmesi hem de diğer yapayy zeka çalışmalarında önmeli bir konuma sahiptir. Verilerin düzensiz, dengesiz veya çok farklı parametrelere sahip olması gibi unsurlar, yapay zeka çalışmalarındaki eğiitm sürecini kötü bir şekilde etkileyebiliyor, bununla beraber performans metriklerine de olumsuz anlamda yansıyor. Biz de bu çalışmamızda, veri ön işleme yapılmış olan ve veri ön işleme yapılmamış olan veri setinin uygulanışı sonucunda, modelin performans metriklerinde ne gibi farkların gerçekleştiğini analiz edip yorumlayacağız.

## Veri Setinin Analizi

![Veri Setinin Analizi](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/5ceae84a-2674-424f-b46d-0ff5e24389d1)

Yukarıda yazdırdığımız tüm bilgileri analiz için kullanabiliriz. Burada dikkatinizi çeken önemli iki durum olmalı. Bunlardan bir tanesi ve daha da önemlisi, sınıfların dengesiz bir şekilde dağılmış olması. Tam da şu olaydan bahsediyorum:
- Class variable (0 or 1)
0 -> 500
1 -> 268
Hasta olmayanların sayısı hasta olanların sayısının iki katına yakın. Bu demek oluyor ki modelimizin eğitim sürecinde bu dengesizlik, performansı ve eğitimi kötü yönde etkileyebilir. Mesela hasta olmayanları yüksek doğruluk oranı ile tespit ederken, hasta olanları düşük doğruluk oranı ile tespit etmesi performans anlamında pek de iyi bir görüntü sağlamayacaktır. Çünkü Precision dediğimiz performans ölçütünün düşük çıkmasına neden olacaktı. Veri setinizde bu şekilde bir dengesizlik var ise performansta bu şekilde anomaliler görmeniz beklenen bir durumdur.

- Bir diğer durum ise, bazı değerlerin 0 olarak kayda geçmiş olması. Birçok sütunda, minimum değer olarak 0 gözlemlenmiş (Plasma glucose concentration, Diastolic blood pressure, Triceps skinfold thickness, 2-Hour serum insulin, ve Body mass index). Bu durumlar, muhtemelen eksik verilerin 0 olarak işaretlendiğini gösteriyor olabilir. Özellikle glukoz, tansiyon ve vücut kitle indeksi gibi ölçümlerde 0 değerlerinin olması gerçekçi değil. Bu değerlerin uygun şekilde işlenmesi gerekiyor. Mesela medyan veya ortalama ile doldurulması gibi gibi..
Şimdi gelin veri setindeki bu dengesizlikleri ve hataları düzeltmeye çalışalım...

### 1-) Kategorilerdeki None Değerler
Features olarak belirlediğimiz değişkende, içeriğinde 0 olması pek de muhtemel olmayan sütunlarımız bulunuyor. Mesela, Diyastolik kan basıncının 0 olması da mümkün değildir. Eğer bir kişinin diyastolik kan basıncı 0 ise, bu kan dolaşımının olmadığı anlamına gelir ki bu hayatta kalmak için imkansızdır. Bundan dolayı Diastolic blood pressure sütununu bu listeye dahil ettik. Listede bulunan diğer sütunlar da bu şekilde eklenmiştir. Sonrasında ise for each dmngüsü ile bu kategorilerin içerisindeki none değerleri yani 0 değerlerini o kategorinin medyanı ile değiştiriyoruz. Rastgele bir sayı yerine medyan koymamızın nedeni ise, veriyi daha dengeli hale getirelim derken daha da beter etmemek. Sizler medyan yerine ortalama da kullanabilirsiniz.

- Gerekli değişiklikleri yapmadan önce:

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/0111b5a7-0320-4891-908d-0d8654a55fe8)

- İlgili değişiklikleri yaptıktan sonra:

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/4cf62027-5977-4954-b42e-67c658b02ce9)

### 2-) Aykırı Değerleri Tespit Ederek Çıkarma
Aykırı değerleri daha iyi gözlemlemek için de her kategori için, box plot olarak bilinen kutu grafiğini detaylı bir şekilde inceliyoruz..

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/5cd26b0e-c963-4102-88b1-9dc70d94c67f)

Burada görüldüğü üzere kutuların dışında kalmış olan kısımlarda yer alan yuvarlaklar bizim aykırı değerlerimizi temsil ediyor. Şimdi bunları Çeyrekler Arası Aralık (IQR) yöntemini kullanarak temizleyelim.
Aykırı verileri datamızdan çıkardıktan sonra şöyle bir kutu grafiğine kavuşuyoruz:

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/94a5497c-9f50-4ae6-85b6-4fc9f4631a09)

Dikkat edersek aykırı verilerin %75-80 civarını temizledik yani tüm aykırı değerleri çıkartmadık. Bunun nedeni ise modelimizin ezberleme olasılığını azaltmak olarak açıklayabiliriz. Sayısı fazla olmamak şartıyla aykırı değerleri veri setimizde bulundurmamız, modelin overfit dediğimiz eğitim verilerini ezberleyerek farklı verilerde saçmalaması probleminin önüne geçecektir. Daha özet bir şekilde açıklamak gerekirse, modelimizin genelleştirilmesine yardımcı olacaktır.

### 3-) Sentetik Veri Üretimi
Şimdi ise dengesiz olan verilerimizi dengeli hale getirmek amacııyla sentetik veri üreteceğiz. Bu işlemi yaparken SMOTE sınıfını kullanacağız. SMOTE sınıfı sentetik şekilde veri üretimi işlemini K-en yakın komşu algoritmasını uygukayarak gerçekleştirmektedir. Rastgele bir k komşusu seçilir. Seçilen örnek ile bu komşu arasındaki her bir özellik farkı hesaplanır. Bu farkın rastgele bir kısmı (0 ile 1 arasında bir sayı ile çarpılarak) seçilen örneğin özelliklerine eklenir. Bu işlem, özellik uzayında, var olan örnek ile seçilen komşusu arasında yeni bir nokta oluşturur. Bu yeni nokta, orijinal veri noktalarına benzer özelliklere sahip sentetik bir örnektir. Bu işlemi de gerçekleştirdikten sonra sınıf sayımızı kontrol ettiğimizde;

Class variable (0 or 1)
- 1 -> 387
- 0 -> 387

Name: count, dtype: int64

Bu sonucu alıyor olmamız gerekiyor. Gördüğünüz üzere arasında 200-300 sample fark varken şu anda her iki sınıfımızın da örnek sayısı eşitlenmiş oldu..

## Gerçekleştirdiğimiz preprocessing işlemlerinin üzerinden geçelim
Bu aşamaya kadar yaptığımız işlemler:

1. Veri setini daha iyi bir şekilde analiz etmek için sütunların sayısal metriklerini ekrana yazdırdık, grafikler çizdirdik.
2. Bu işlemlerden olmaması gereken yerde ilgiisiz verileri gözlemledik. 0 olmaması gereken bir sütunda 0 olması gibi.
3. Veri setimizde aykırı değerleri belirledik ve çıkarttık.
4. Verideki dengesiz durumunu dengelemek amacıyla az olan sınıftan sentetik veriler ürettik.
5. Son olarak normalizasyon işlemi gerçekleştirdik. (İlerleyen aşamada..)
Bundan sonraki aşamalarda ise modeli oluşturup eğitim işlememizi gerçekleştireceğiz.

## Uygulayacağımız Algoritmalara Göz Atalım
### 1-) Naive Bayes Algoritması
Naive bayes algoritması olasılığa dayalı bir sınıflandırma yöntemidir. Algoritma, bir veri için tüm olasılıkları hesaplar ve olasılık değeri en yüksek çıkan sonuca göre diğer verilerin sınıflandırmasını yapar. Eğitim sürecinde ise, eğitimi aldığı veri üzerindeki her bir sınıfın frekansını, sınıflar içerisindeki özelliklerin varyans ve ortalamalarını hesaplayarak öğrenir. Tahmin yaparken bu faktörleri göz önünde bulundurarak olas bir cevap üretir.

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/90639ba1-2ed1-4a14-a1e4-b486c8d7780a)

Buradaki kavramları daha detaylı bir şekilde açıklayacak olursak:
- P( A | B ) = B olayı gerçekleştiğinde A olayının gerçekleşme olasılığı
- P( A ) = A olayının gerçekleşme olasılığı
- P( B | A ) = A olayı gerçekleştiğinde B olayının gerçekleşme olasılığı
- P( B ) = B olayının gerçekleşme olasılığı

Şeklinde özetleyebiliriz.

### 2-) K-En Yakın Komşu Algoritması (KNN)
K-En yakın komşu algoritması hem sınıflandırma hemde regresyon problemlerini çözmek için kullanılan,basit ve uygulaması kolay bir denetimli öğrenme algoritmasıdır. Bu algoritma, komşularına bakarak tahminlemede bulunan bir yapıa sahiptir. KNN algoritmasında, benzer olan şeyler birbirine yakındır varsayımı geçerlidir. Aşağıdaki resimde bunu çok daha net bir şekilde gözlemleyebilirsiniz..

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/365050c9-d8eb-4956-bd25-0426c1aa57d5)

Burada aklımıza bir soru gelmiş olabilir. Ölçülen mesafe nedir? Evet Machine learning algoritmalarında farklı mesafe ölçümleri kullanılmaktadır. Temel olarak Öklid mesafesi (Euclidean distance) kullanılsa da, daha farklı ölçümlerde mevcuttur. Biz çalışmamızda öklid mesafesinden yararlanarak modelimizi geliştireceğiz. Diğer mesafe ölçümlerini de detaylı bir şekilde incelemek isterseniz: https://medium.com/machine-learning-t%C3%BCrkiye/machine-learning-mesafeleri-8ac88ca393 bu linkten faydalanabilirsiniz.

### 3-) Çok Katmanlı Algılayıcı (MLP)
Çok katmanlı algılayıcılar, yapay sinir ağlarının vazgeçilmezi olan bir yapıdır. Temelde XOR problemini çözmek için yapılan çalışmalar sonucu ortaya çıkmış olan MLP, insan beyninin bilgi işleme şeklini taklit etmeye çalışan matematiksel bir model olmasının yanı sıra yapay sinir ağlarının en temel ve yaygın formlarından biridir. MLP, birbirine bağlı nöronlardan (yapay sinir hücrelerinden) oluşan birden fazla katmana sahiptir. Bu katmanlar genellikle

1. Giriş Katmanı: Verilerin model tarafından alındığı ilk noktadır.
2. Gizli Katman veya Katmanlar: Her biri bir dizi ağırlık ve bias içeren ve karmaşık özelliklerin öğrenildiği katmanlardır.
3. Çıkış Katmanı: Sonuçların (tahminlerin) üretildiği katmandır.

şeklinde özetlenebilir.

3 katmanla birlikte 2 bölümden oluşan bir yapıya sahiptir. Bu bölümlerden ilki "İleri Doğru Hesaplama" iken diğer ise "Geri Doğru Hesaplamadır". Başlangıçta rastgele atadığımız ağırlıklar ve bias ileri doğru hesaplamadan sonra ulaşılan tahmin değerlerine göre, geri doğru hesaplama ile her seferinde güncellenir ve böylelikle her seferinde hata oranımızı azaltmış oluruz. Peki bu bias ve ağırlıklar nedir? Her bağlantı, bir ağırlığa (verinin önemini belirleyen) ve her nöron, bir biasa (aktivasyon eşiğini ayarlayan) sahiptir.

MLP sürecinde bir diğer önemli yapı ise aktivasyon fonksiyonlarıdır. Aktivasyon fonksiyonları ile her nöronun çıktısı, genellikle bir aktivasyon fonksiyonu (örneğin, sigmoid, ReLU) tarafından işlenir. Bu fonksiyonlar, nöronların lineer olmayan karmaşık örüntüleri öğrenmesini sağlar. Yani günlük hayata daha uygun bir şekilde modelin genelleştirilmesine katkıda bulunur.

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/603d206f-b639-4bc9-a5a4-6679b506d445)

### 4-) Destek Vektör Mekanizması (SVM)
Destek Vektör Makineleri (Support Vector Machine) genellikle sınıflandırma problemlerinde kullanılan gözetimli öğrenme yöntemlerinden biridir. Bir düzlem üzerine yerleştirilmiş noktaları ayırmak için bir doğru çizer.

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/3a4a7997-7b9b-45c8-85a5-a4b121c37f51)

Tabloda siyahlar ve beyazlar olmak üzere iki farklı sınıf var. Sınıflandırma problemlerindeki asıl amacımız gelecek verinin hangi sınıfta yer alacağını karar vermektir. Bu sınıflandırmayı yapabilmek için iki sınıfı ayıran bir doğru çizilir ve bu doğrunun ±1'i arasında kalan yeşil bölgeye Margin adı verilir. Margin ne kadar geniş ise iki veya daha fazla sınıf o kadar iyi ayrıştırılır. Bu doğrunun, iki sınıfının noktaları için de maksimum uzaklıkta olmasını amaçlar. SVM veri setinin lineer olup olmamasına göre ikiye ayrılmaktadır. Biz çalışmamızda Doğrusal (Lineer) SVM kullanacağız.
1. Doğrusal (Lineer) SVM: Doğrusal SVM, doğrusal olarak ayrılabilen veri kümeleri için kullanılır. Aşağıdaki görselden de anlaşılacağı üzere düz bir çizgi iki grubu etkili bir şekilde ayırıp sınıflandırma yapabiliyor.

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/ab19ed12-6ead-43ac-9ae0-833ba5b8ff68)

2. Doğrusal Olmayan (Nonlinear) SVM: Doğrusal Olmayan SVM ise düz bir çizgi ile ayrılamayan veri kümeleri için kullanılmaktadır. Doğrusal bir hiperdüzlem kullanılamadığı için “kernel trick” denilen yapılar kullanılır. Bu sayede yüksek sınıflandırma oranı elde edilir. Kernel trick çeşitleri ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’ ve ‘precomputed’ olmak üzere beş çeşittir ancak en çok kullanılanlar ‘poly’ ile ‘rbf’ tir.

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/a6199175-018a-4c35-9f72-dbc265a04726)

## Eğitimler Sonucunda Modellerin Performans Metriklerinin Değerlendirilmesi
4 farklı algoritmanın eğitimi sonucunda doğruluk oranları 0.7-0.8 aralığında gidip geldi.. Bu algoritmalardan en yüksek doğruluk oranına sahip olan sistem sizin de tahmin edeceğiniz üzere çok katmanlı algılayıcı olarak adlandırdığımız ve yapay sinir ağlarının temel taşı olarak bilinen MLP algoritması oldu. Gizli katmanlarının sayısının ya da katmanlardaki nöron sayılarının arttırılması/azalması gibi faktörlerin de performansında etkili olduğu bilinen bu model, çalışmamızda en yakın takipçisine 0.3 fark attı (ROC Eğrisi baz alınmıştır). Tüm algoritmaların performanslarına bir göz atmadan önce, kullandığımız ölçütlerin neleri ifade ettiğini özetlemek isterim:

### 1. Confusion Matrix
Bu tablo modelin tahmin sonuçları ile gerçek sonuçları karşılaştırılarak dört farklı durumu özetler. Bunlar:
- True Positive (TP): Gerçek pozitif örneklerin doğru olarak pozitif tahmin edilmesi.
- True Negative (TN): Gerçek negatif örneklerin doğru olarak negatif tahmin edilmesi.
- False Positive (FP): Gerçek negatif örneklerin yanlışlıkla pozitif tahmin edilmesi.
- False Negative (FN): Gerçek pozitif örneklerin yanlışlıkla negatif tahmin edilmesi.

Confusion matrix kullanılarak çeşitli performans metrikleri hesaplanabilir. Bunlar:
- Accuracy (Doğruluk):

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/99489ebf-79be-4121-944f-6cca04c193b2)

Doğru tahminlerin tüm tahminlere oranını verir.
- Precision (Kesinlik):

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/65796e14-40f4-45ac-b535-8b894d2b3932)

Pozitif tahminlerin ne kadarının doğru olduğunu gösterir.
- Recall (Duyarlılık):

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/69498474-e2c9-4248-a770-b696bb58eb9d)

Gerçek pozitiflerin ne kadarının doğru tahmin edildiğini gösterir.
- Specificity (Özgüllük):

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/811a5a73-2f7a-44aa-ad1d-a625d6d59c79)

Gerçek negatiflerin ne kadarının doğru tahmin edildiğini gösterir.
- F1 Score:

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/b61d54a8-f54e-4fd8-953b-1cc2e35ce0f8)

Precision ve Recall'un harmonik ortalamasıdır.

Bunlar haricinde confusion matrix ile model analizi de yapabiliriz:
- Yüksek False Positive: Model, negatif örnekleri pozitif olarak sınıflandırma eğilimindedir. Bu, sahte alarmlara yol açabilir.
- Yüksek False Negative: Model, pozitif örnekleri negatif olarak sınıflandırma eğilimindedir. Bu, önemli pozitif örneklerin gözden kaçmasına neden olabilir.

gibi gibi..
### 2. ROC Eğrisi
ROC eğrisi, modelin doğru pozitif oranı (True Positive Rate, TPR) ve yanlış pozitif oranı (False Positive Rate, FPR) arasındaki dengeyi gösterir. Ayrıca modelin farklı eşik değerlerinde (thresholds) nasıl performans gösterdiğini anlamak için de kullanılabilir. ROC eğrisi FPR ekseninde (x ekseni) ve TPR ekseninde (y ekseni) çizilir. Farklı eşik değerleri kullanılarak, modelin FPR ve TPR değerleri hesaplanır ve bu değerler grafikte bir eğri oluşturur. Hemen FPR ve TPR kavramlarının ne olduğunu da açıklayalım.
- False Positive Rate (FPR):

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/14a03d1c-1dda-4cd2-9a6e-c70713841e1f)

Gerçek negatif örneklerin yanlış pozitif olarak tahmin edilme oranıdır.
- True Positive Rate (TPR):

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/701bf488-1296-4c15-adc1-981be5b802df)

Gerçek pozitif örneklerin doğru tahmin edilme oranıdır.

Peki biz bu eğriden nasıl performans ölçütü çıkaracağız? Onu da açıklayalım; ROC eğrisi altındaki alan (AUC), modelin genel performansını özetleyen tek bir değer sağlar. AUC değeri 0.5 ile 1 arasında değişir:
- AUC = 0.5: Model tamamen rastgele tahmin yapıyor demektir.
- AUC = 1: Model mükemmel tahminler yapıyor demektir.

AUC değeri ne kadar yüksekse, modelin sınıflandırma performansı o kadar iyidir.

### Algoritmaların Performans Metrikleri
#### 1-) MLP Allgoritması
- Confusion Matrix ve Performans Ölçütleri:
![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/47cd30eb-efd2-4bbd-b9cd-3b5ca09c1ce6)
- Accuracy: 0.8025751072961373
- Precision: 0.8125
- Recall/Sensitivity: 0.7844827586206896
- Specificity: 0.8205128205128205
- F1 Score: 0.7982456140350878
- ROC Eğrisi:

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/163b8fa1-8063-45af-ace6-638d87ac4da7)

#### 2-) KNN Algoritması
- Confusion Matrix ve Performans Ölçütleri:
![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/c3154643-fd05-4582-b070-37e8836a7cbb)
- Accuracy: 0.7682403433476395
- Precision: 0.7094594594594594
- Recall/Sensitivity: 0.9051724137931034
- Specificity: 0.6324786324786325
- F1 Score: 0.7954545454545455
- ROC Eğrisi:

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/a6cbff5e-acea-4bff-840a-aafb0b716c69)

#### 3-) SVM Algoritması
- Confusion Matrix ve Performans Ölçütleri:
![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/70cd65d1-ac95-4ee2-83fe-1f6645bc5c3f)
- Accuracy: 0.7424892703862661
- Precision: 0.71875
- Recall/Sensitivity: 0.7931034482758621
- Specificity: 0.6923076923076923
- F1 Score: 0.7540983606557378
- ROC Eğrisi:

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/dfead3f8-125f-4891-be53-2cadcd2f779a)

#### 4-) Naive Bayes Algoritması
- Confusion Matrix ve Performans Ölçütleri:
![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/42c6cdce-9611-4454-83e5-6d52892c4c24)
- Accuracy: 0.7296137339055794
- Precision: 0.71900826446281
- Recall/Sensitivity: 0.75
- Specificity: 0.7094017094017094
- F1 Score: 0.7341772151898734
- ROC Eğrisi:

![image](https://github.com/emircanakyuz/Makine_Ogrenmesi_Algoritmalarinin_Uygulanisi/assets/95855820/a1c058bf-f4a4-4271-8b85-fda71efe1587)
