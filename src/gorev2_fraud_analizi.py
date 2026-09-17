"""
Görev 2: Fraud (Dolandırıcılık) Tespit Modeli - Confusion Matrix Analizi
-----------------------------------------------------------------------------
Banka işlemlerinde dolandırıcılığı yakalamak için kurulan, %90.5 accuracy
ile "başarılı" bulunup canlıya alınan ancak iş biriminin başarısız
bulduğu modelin confusion matrix'i üzerinden Accuracy, Precision, Recall,
F1 skorları hesaplanır ve veri bilimi ekibinin gözden kaçırdığı durum
yorumlanır.

Kullanım:
    python src/gorev2_fraud_analizi.py
"""

from pathlib import Path

import pandas as pd

from metrikler import karisiklik_matrisi_ciz, metrikleri_hesapla, sonuclari_yazdir

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

# PDF'de verilen confusion matrix değerleri (Gerçek Fraud=1 / Non-Fraud=0)
TP, FN, FP, TN = 5, 5, 90, 900

YORUM = """
YORUM - Veri Bilimi Ekibinin Gözden Kaçırdığı Durum:

Veri seti ciddi şekilde dengesizdir (class imbalance): 1000 işlemin
sadece 10 tanesi (%1) gerçek dolandırıcılık (fraud) vakasıdır. Böyle
dengesiz bir veri setinde Accuracy metriği tek başına yanıltıcıdır;
her işlemi "Non-Fraud" olarak tahmin eden saf (naive) bir model bile
%99 accuracy elde edebilirdi. Ekip, modeli canlıya almadan önce
yalnızca genel Accuracy'ye (%90.5) bakarak modeli "başarılı" olarak
değerlendirmiş, ancak asıl kritik metrikleri -özellikle Recall'ı-
göz ardı etmiştir:

  - Recall (%50)    : Model, gerçek dolandırıcılık vakalarının sadece
    yarısını yakalayabilmektedir; 10 fraud işleminden 5'i (FN) kaçmıştır.
  - Precision (%5.3): Model "Fraud" dediğinde bu tahminin doğru çıkma
    ihtimali çok düşüktür; 95 fraud alarmından sadece 5'i gerçektir,
    90'ı yanlış alarmdır (FP).

Sonuç olarak model hem çok sayıda gerçek dolandırıcılığı kaçırmakta
hem de iş birimini gereksiz yere çok sayıda yanlış alarmla meşgul
etmektedir. Dengesiz veri setlerinde başarı ölçütü olarak Accuracy
yerine Precision, Recall, F1 skoru (ve mümkünse iş maliyetine göre
ağırlıklandırılmış metrikler) esas alınmalıydı.
""".strip()


def main() -> None:
    metrikler = metrikleri_hesapla(tp=TP, fp=FP, fn=FN, tn=TN)
    sonuclari_yazdir(metrikler, "GÖREV 2 SONUÇLARI (Fraud Tespit Modeli)")

    print("\n" + "-" * 60)
    print(YORUM)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrikler]).to_csv(OUTPUT_DIR / "gorev2_metrikler.csv", index=False)
    karisiklik_matrisi_ciz(
        tp=TP, fp=FP, fn=FN, tn=TN,
        pozitif_etiket="Fraud (1)", negatif_etiket="Non-Fraud (0)",
        baslik="Görev 2: Fraud Modeli Confusion Matrix",
        kayit_yolu=OUTPUT_DIR / "gorev2_confusion_matrix.png",
    )
    print(f"\nÇıktılar '{OUTPUT_DIR}' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
