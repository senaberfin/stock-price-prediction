# Video Metni: "PyTorch ile Hisse Senedi Fiyat Tahmini — LSTM vs GRU"
*(Yaklaşık 5 dakikalık konuşma metni)*

---

## Giriş (0:00 – 0:40)

Merhaba! Bu videoda, PyTorch kullanarak hisse senedi fiyat tahmini yapan bir
makine öğrenmesi projesini nasıl geliştirdiğimi anlatacağım. Temel ML
bilgim vardı — regresyon nedir, eğitim/test ayrımı nedir gibi kavramları
biliyordum — ama bu projeyi yaparken gerçekten çok şey öğrendim. Bu videoda
hem süreci hem de yolda öğrendiklerimi paylaşacağım, sonunda da bu projeyi
nasıl GitHub'a yüklediğimi göstereceğim.

## Projenin Amacı (0:40 – 1:10)

Amaç basit görünüyordu: geçmiş kapanış fiyatlarına bakarak yarının fiyatını
tahmin eden bir model kurmak. Ama bunu yaparken iki farklı tekrarlayan sinir
ağı mimarisini karşılaştırmak istedim: **LSTM** ve **GRU**. İkisi de zaman
serisi verisiyle çalışmak için tasarlanmış, ama iç yapıları farklı. Ben de
"hangisi gerçekten daha iyi, yoksa aralarında pratikte fark var mı?" sorusuna
kendi verimle cevap aramak istedim.

## Veri Hazırlığı Aşamasında Öğrendiklerim (1:10 – 2:00)

İlk büyük öğrenme burada oldu. Ham fiyat verisini doğrudan modele veremezsiniz.
Önce **MinMax ölçekleme** ile fiyatları -1 ile 1 arasına sıkıştırmam
gerektiğini öğrendim, çünkü LSTM ve GRU içinde kullanılan tanh ve sigmoid
fonksiyonları bu aralıkta çok daha iyi çalışıyor. Sonra **sliding window**,
yani kayan pencere yöntemini öğrendim: son 20 günün fiyatını alıp 21. günü
tahmin ettiriyorsunuz, sonra pencereyi bir gün kaydırıp tekrar aynısını
yapıyorsunuz. Basit bir fikir ama zaman serisini nasıl "denetimli öğrenme"
problemine çevirdiğinizi gerçekten anlamamı sağladı. Ayrıca veriyi karıştırmadan,
**kronolojik sırayla** eğitim ve test olarak ayırmam gerektiğini öğrendim —
zaman serisinde geleceği görüp geçmişi tahmin etmek gibi bir hata yapmamak
çok önemli.

## Model Kurma ve Eğitim (2:00 – 2:50)

PyTorch'ta `nn.Module` sınıfından LSTM ve GRU modellerini sıfırdan yazdım.
Burada en çok şaşırdığım şey, LSTM'in hem "hidden state" hem "cell state"
tutması, GRU'nun ise bunları tek bir durumda birleştirmesiydi. Yani GRU daha
az parametre kullanıyor, bu yüzden genelde daha hızlı eğitiliyor. İki modeli
de tamamen aynı hiperparametrelerle — aynı katman sayısı, aynı gizli boyut,
aynı epoch sayısı — eğittim ki karşılaştırma adil olsun. Kayıp fonksiyonu
olarak MSE, optimizasyon için de Adam kullandım. Eğitim döngüsünü yazarken
`loss.backward()` ve `optimizer.step()` adımlarının aslında nasıl çalıştığını
çok daha iyi kavradım; önceden bunlar bana "kara kutu" gibi geliyordu.

## Değerlendirme ve Beklenmedik Bir Ders (2:50 – 3:50)

Modelleri test verisinde değerlendirirken MSE ve RMSE hesapladım ve
tahminleri gerçek fiyat birimine geri çevirmem gerektiğini öğrendim — yoksa
sonuçlar ölçeklenmiş, anlamsız sayılar olarak kalıyor. Ama asıl önemli ders
şuydu: sadece LSTM ve GRU'yu karşılaştırmak yetmiyor. Basit bir **naif
tahmin** ile de karşılaştırmam gerekiyordu — yani "yarının fiyatı bugünkü
fiyata eşittir" diyen çok basit bir yöntemle. Ve gördüm ki, borsa verisi
neredeyse rastgele yürüyüşe benzediği için, bazen bu basit yöntem bile
karmaşık modellerle yarışabiliyor. Bu benim için gerçekten gözü açan bir
andı: düşük hata payı her zaman "model bir şey öğrendi" anlamına gelmiyor.
Bu yüzden projeye kritik bir bakış açısı da kattım ve "AI Snake Oil" gibi
kaynaklara referans verdim.

## Projeyi Toparlama (3:50 – 4:20)

Son olarak projeyi sadece çalışan bir not defteri olarak bırakmadım. Kodları
`src/` klasöründe modüler fonksiyonlara ayırdım, bunlar için birim testleri
yazdım, GitHub Actions ile otomatik test/lint kontrolü kurdum. Bu kısım bana
"iyi bir ML projesinin sadece doğru tahmin yapmakla değil, tekrarlanabilir,
test edilebilir ve belgelenmiş olmakla da ilgili olduğunu" öğretti.

## Kapanış (4:20 – 5:00)

Özetle, bu proje bana zaman serisi verisini hazırlamayı, LSTM ve GRU
arasındaki gerçek farkı, model değerlendirmede naif bir referans noktasının
önemini ve bir projeyi GitHub'a düzgün şekilde nasıl hazırlayacağımı
öğretti. Şimdi bu projeyi nasıl GitHub'a yüklediğimi göstereceğim — birazdan
ekranda adım adım izleyebilirsiniz. İzlediğiniz için teşekkürler, kodun
tamamına GitHub bağlantısından ulaşabilirsiniz!
