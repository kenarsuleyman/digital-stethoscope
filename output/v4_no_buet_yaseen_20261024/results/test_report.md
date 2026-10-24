# Test Seti Değerlendirme Sonuçları

| Metrik | Değer |
|--------|-------|
| Doğruluk (Accuracy) | 0.7448 |
| Duyarlılık (Sensitivity/Recall) | 0.5283 |
| Özgüllük (Specificity) | 0.8686 |
| Kesinlik (Precision) | 0.6969 |
| F1 Skoru | 0.6010 |
| AUC-ROC | 0.7551 |
| Ortalama Kesinlik (AP) | 0.7197 |

## Karışıklık Matrisi

|  | Tahmin: Normal | Tahmin: Anormal |
|--|----------------|-----------------|
| Gerçek: Normal | 2493 | 377 |
| Gerçek: Anormal | 774 | 867 |

- Toplam test örneği: 4511
- Pozitif (Anormal): 1641
- Negatif (Normal): 2870
- Eşik değeri: 0.5
