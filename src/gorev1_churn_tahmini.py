"""
Görev 1: Churn Tahmin Modeli - Confusion Matrix ve Sınıflandırma Metrikleri
-----------------------------------------------------------------------------
Müşterinin churn olup olmama durumunu tahminleyen bir sınıflandırma
modelinin 10 test gözlemi için ürettiği olasılık tahminleri kullanılarak;
eşik değeri 0.5 alınarak confusion matrix oluşturulur ve Accuracy,
Precision, Recall, F1 skorları hesaplanır.

Kullanım:
    python src/gorev1_churn_tahmini.py
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

from metrikler import karisiklik_matrisi_ciz, metrikleri_hesapla, sonuclari_yazdir

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "churn_test_verisi.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
ESIK_DEGERI = 0.5


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Eşik değerine göre sınıf tahmini üret: olasilik >= 0.5 ise 1 (Churn)
    df["tahmin_sinif"] = (df["olasilik_tahmini"] >= ESIK_DEGERI).astype(int)

    print("Tahmin Tablosu:\n")
    print(df.to_string(index=False))
    print()

    # labels=[1, 0] -> cm[0,0]=TP, cm[0,1]=FN, cm[1,0]=FP, cm[1,1]=TN
    cm = confusion_matrix(df["gercek_deger"], df["tahmin_sinif"], labels=[1, 0])
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    metrikler = metrikleri_hesapla(tp=tp, fp=fp, fn=fn, tn=tn)
    sonuclari_yazdir(metrikler, f"GÖREV 1 SONUÇLARI (Eşik Değeri = {ESIK_DEGERI})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrikler]).to_csv(OUTPUT_DIR / "gorev1_metrikler.csv", index=False)
    karisiklik_matrisi_ciz(
        tp=tp, fp=fp, fn=fn, tn=tn,
        pozitif_etiket="Churn (1)", negatif_etiket="Non-Churn (0)",
        baslik="Görev 1: Churn Modeli Confusion Matrix",
        kayit_yolu=OUTPUT_DIR / "gorev1_confusion_matrix.png",
    )
    print(f"\nÇıktılar '{OUTPUT_DIR}' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
