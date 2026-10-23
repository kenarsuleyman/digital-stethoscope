# Test Seti Değerlendirme Sonuçları

| Metrik | Değer |
|--------|-------|
| Doğruluk (Accuracy) | 0.8918 |
| Duyarlılık (Sensitivity/Recall) | 0.8032 |
| Özgüllük (Specificity) | 0.9425 |
| Kesinlik (Precision) | 0.8887 |
| F1 Skoru | 0.8438 |
| AUC-ROC | 0.9456 |
| Ortalama Kesinlik (AP) | 0.9288 |

## Karışıklık Matrisi

|  | Tahmin: Normal | Tahmin: Anormal |
|--|----------------|-----------------|
| Gerçek: Normal | 2705 | 165 |
| Gerçek: Anormal | 323 | 1318 |

- Toplam test örneği: 4511
- Pozitif (Anormal): 1641
- Negatif (Normal): 2870
- Eşik değeri: 0.5
