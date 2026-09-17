"""
Sınıflandırma modelleri için ortak metrik hesaplama ve görselleştirme
fonksiyonları. Hem Görev 1 hem de Görev 2 tarafından kullanılır.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def metrikleri_hesapla(tp: int, fp: int, fn: int, tn: int) -> dict:
    """
    Confusion matrix bileşenlerinden (TP, FP, FN, TN) Accuracy, Precision,
    Recall ve F1 skorlarını hesaplar.

        Accuracy  = (TP + TN) / n
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        F1        = 2 * Precision * Recall / (Precision + Recall)
    """
    n = tp + fp + fn + tn
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "gozlem_sayisi": n,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def sonuclari_yazdir(metrikler: dict, baslik: str) -> None:
    """Hesaplanan metrikleri konsola okunabilir biçimde yazdırır."""
    print("=" * 60)
    print(baslik)
    print("=" * 60)
    print(f"TP={metrikler['TP']}  FP={metrikler['FP']}  "
          f"FN={metrikler['FN']}  TN={metrikler['TN']}  (n={metrikler['gozlem_sayisi']})")
    print(f"Accuracy  : {metrikler['Accuracy']:.4f}")
    print(f"Precision : {metrikler['Precision']:.4f}")
    print(f"Recall    : {metrikler['Recall']:.4f}")
    print(f"F1 Skoru  : {metrikler['F1']:.4f}")


def karisiklik_matrisi_ciz(tp: int, fp: int, fn: int, tn: int,
                            pozitif_etiket: str, negatif_etiket: str,
                            baslik: str, kayit_yolu: Path) -> None:
    """
    Confusion matrix'i ısı haritası (heatmap) olarak çizip PNG dosyasına
    kaydeder. Satırlar gerçek değeri, sütunlar model tahminini gösterir.
    """
    matris = np.array([[tp, fn],
                        [fp, tn]])
    etiketler = [pozitif_etiket, negatif_etiket]

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(matris, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(etiketler)
    ax.set_yticklabels(etiketler)
    ax.set_xlabel("Model Tahmini")
    ax.set_ylabel("Gerçek Değer")
    ax.set_title(baslik)

    esik = matris.max() / 2 if matris.max() > 0 else 0
    for i in range(2):
        for j in range(2):
            deger = matris[i, j]
            renk = "white" if deger > esik else "black"
            ax.text(j, i, f"{deger}", ha="center", va="center",
                     color=renk, fontsize=15, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    kayit_yolu.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(kayit_yolu, dpi=150)
    plt.close(fig)
