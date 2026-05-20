# Kalp Sesi Sınıflandırma Modeli — Claude Code Geliştirme Planı

Tezinin 3.1 ve 3.4 bölümlerindeki metodolojiyi birebir uygulayacak şekilde, Claude Code ile adım adım ilerleyebileceğin bir geliştirme planı. Her aşamayı ayrı bir prompt/görev olarak Claude Code'a verebilirsin.

---

## Aşama 0 — Proje İskeleti ve Ortam Kurulumu

İlk olarak temiz bir proje yapısı kur. Claude Code'a şunu yaptır:

- `heart-sound-classifier/` kök dizini altında `data/raw`, `data/processed`, `data/segments`, `notebooks`, `src`, `models`, `logs`, `configs` klasörleri.
- `requirements.txt`: `numpy`, `scipy`, `librosa`, `soundfile`, `tensorflow==2.15.*`, `scikit-learn`, `matplotlib`, `pandas`, `tqdm`, `pyyaml`.
- `configs/config.yaml`: örnekleme frekansı (4000), segment uzunluğu (5 s), bant geçiren sınırlar (25–900 Hz), Mel parametreleri (n_fft=512, hop=256, n_mels=64), eğitim hiperparametreleri. Tüm sabitleri buradan oku — kod içine gömme.
- `.gitignore`, `README.md` ve `Makefile` (her aşama için ayrı hedef: `make preprocess`, `make train`, `make convert`).

---

## Aşama 1 — Veri Setlerinin İndirilmesi ve Düzenlenmesi

`src/data/download.py` betiği:

- PhysioNet/CinC 2016 training set (a–e) ve PASCAL Dataset-B'yi `data/raw/` altına indirsin.
- Her iki veri seti için bir `metadata.csv` üretsin: `file_path`, `dataset_source`, `original_sr`, `original_label`, `binary_label` (Normal=0, Anormal=1), `patient_id`. PASCAL'ın üç sınıfını ikiliye indir: Üfürüm + Ekstra Sistol → Anormal.
- PASCAL Dataset-A'yı **dahil etme** (tezinde gerekçesi belirtilmiş).

---

## Aşama 2 — Yeniden Örnekleme ve Segment Bölümleme

`src/preprocessing/resample_segment.py`:

- PhysioNet kayıtlarını `scipy.signal.resample_poly(up=2, down=1)` ile 2000 Hz → 4000 Hz'e yükselt. PASCAL'a dokunma.
- Tüm kayıtları 5 s = 20.000 örneklik segmentlere böl. Son artık parça 5 s'den kısaysa at.
- Her segmenti `data/segments/{patient_id}_{rec_id}_{seg_idx}.wav` olarak kaydet ve `segments_metadata.csv` üret (segment yolu + binary_label + patient_id).

> **Önemli:** `patient_id` alanını koru — Aşama 6'daki hasta bazlı bölümleme için kritik.

---

## Aşama 3 — Veri Temizleme ve Filtreleme

`src/preprocessing/clean_filter.py`:

1. **Enerji eşikleme:** Her segmentin RMS'ini hesapla. Tüm segmentlerin medyan RMS'ini bul. `0.1 × medyan`'ın altındakileri ve `10 × medyan`'ın üstündekileri at.
2. **Bant geçiren filtre:** 4. dereceden Butterworth, 25–900 Hz, `scipy.signal.butter(..., output='sos')` + `sosfiltfilt` (sıfır faz).
3. **Normalizasyon:** Her segmenti kendi `max(abs)` değerine bölerek [−1, +1] aralığına ölçekle.
4. Temizlenmiş segmentleri `data/processed/` altına kaydet, elenen segmentleri ayrı bir log dosyasına yaz.

---

## Aşama 4 — Mel-Spektrogram Dönüşümü

`src/preprocessing/mel_spectrogram.py`:

- `librosa.feature.melspectrogram` ile parametreleri Çizelge 3.2'den uygula (sr=4000, n_fft=512, hop_length=256, n_mels=64, fmin=25, fmax=900).
- `librosa.power_to_db` ile dB'ye çevir.
- [0, 1] min-max normalizasyonu.
- Tek kanallı matrisi 3 kez yığarak (np.stack 3 kez) RGB formatına getir, 224×224'e yeniden boyutlandır (`tf.image.resize`).
- Çıktıları `.npy` veya `tf.data.Dataset` cache'i olarak kaydet — eğitim sırasında her seferinde yeniden hesaplama.

---

## Aşama 5 — Veri Artırma Pipeline'ı

`src/preprocessing/augmentation.py` — **yalnızca eğitim setine, on-the-fly:**

- **Zaman kaydırma:** ham PCG üzerinde ±1000 örnek (sıfır padding ile kaydır), sonra Mel'i yeniden hesapla.
- **Zaman maskeleme:** Mel matrisinde 5–15 ardışık zaman adımını sıfırla.
- **Frekans maskeleme:** 2–8 ardışık Mel bandını sıfırla.
- **Rastgele genlik ölçekleme:** tüm Mel matrisini U(0.8, 1.2)'den çekilen katsayıyla çarp.

`tf.data.Dataset.map` içinde `tf.py_function` ile sar.

---

## Aşama 6 — Hasta Bazlı Veri Bölümleme

`src/data/split.py`:

- `sklearn.model_selection.GroupShuffleSplit` kullanarak `patient_id`'ye göre %70/%15/%15 eğitim/doğrulama/test böl.
- Sınıf dağılımını her split için yazdır.
- Sınıf ağırlıklarını `sklearn.utils.class_weight.compute_class_weight('balanced', ...)` ile hesapla, `configs/class_weights.json` olarak kaydet.

---

## 🔄 Yerel Makineden Google Colab'a Geçiş

Aşama 0–6 yerel makinende (Claude Code ile) tamamlandıktan sonra **Aşama 7, 8 ve 10 için Google Colab'a (NVIDIA T4 GPU) geçilecek.** Tezinde de eğitim ortamı olarak Colab + T4 GPU belirtilmiş (toplam ~25 dk eğitim süresi).

**Geçiş için yapılacaklar:**

1. `data/processed/` klasörünü (veya cache'lenmiş `.npy` / `tf.data` dosyalarını) ve `configs/`, `splits/` dosyalarını `.zip` olarak sıkıştır.
2. Google Drive'a yükle (örn. `MyDrive/heart-sound-classifier/`).
3. Colab notebook'unda Drive'ı mount et:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !unzip -q /content/drive/MyDrive/heart-sound-classifier/processed.zip -d /content/data
   ```
4. Runtime → Change runtime type → **T4 GPU** seç.
5. `requirements.txt`'i Colab'a kur (TensorFlow zaten kurulu, eksikleri tamamla).
6. Eğitim bitince üretilen Keras modelini (`.h5` veya `SavedModel`) ve TFLite modelini Drive'a geri yaz.

> **İpucu:** Colab oturumu kopabileceği için checkpoint'leri her epoch sonunda Drive'a yazdır (`ModelCheckpoint` callback'inin `filepath`'ini Drive yoluna ver). Aksi halde 25 dakikalık eğitim ortasında bağlantı kopması her şeyi sıfırlar.

---

## Aşama 7 — Model Mimarisi *(Google Colab)*

`src/models/build_model.py`:

```python
MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
→ GlobalAveragePooling2D
→ Dropout(0.3)
→ Dense(1, activation='sigmoid')
```

İki fonksiyon yaz:
- `build_stage1_model()` — tüm base dondurulmuş
- `unfreeze_for_stage2(model, n_blocks=4)` — son 4 inverted residual bloğu çöz

---

## Aşama 8 — İki Aşamalı Eğitim *(Google Colab — NVIDIA T4 GPU)*

`src/training/train.py` (Colab notebook olarak `notebooks/train_colab.ipynb` şeklinde de hazırlanabilir):

> **Donanım:** NVIDIA T4 GPU (16 GB VRAM). Beklenen toplam süre: ~25 dakika (Aşama 1: ~5 dk, Aşama 2: ~20 dk).

- **Aşama 1:** Adam, lr=1e-3, 10 epoch, base donmuş.
- **Aşama 2:** Son 4 blok çözük, Adam, lr=1e-5, 30 epoch.
- **Callback'ler:** `ModelCheckpoint` (en iyi val_loss), `EarlyStopping(patience=7, restore_best_weights=True)`, `ReduceLROnPlateau(factor=0.5, patience=3)`, `TensorBoard(log_dir='logs/')`, `CSVLogger`.
- `class_weight` parametresini `model.fit`'e geçir.
- **Kayıp:** `binary_crossentropy`. **Metrikler:** `accuracy`, `AUC`, `Precision`, `Recall`.
- Eğitim/doğrulama loss ve accuracy eğrilerini `figures/training_curves.png` olarak kaydet.

---

## Aşama 9 — Test Seti Değerlendirmesi

`src/evaluation/evaluate.py`:

- Test seti üzerinde tahmin yap (eşik = 0.5).
- Hesapla: accuracy, sensitivity (recall), specificity, precision, F1, AUC-ROC.
- Confusion matrix, ROC eğrisi ve precision-recall eğrisini çiz.
- Sonuçları `results/test_metrics.json` ve `results/test_report.md` olarak kaydet — bunları doğrudan tezin Bölüm 4'üne aktarabilirsin.

---

## Aşama 10 — TFLite Dönüşümü ve Doğrulama *(Google Colab)*

`src/deployment/convert_tflite.py`:

- `tf.lite.TFLiteConverter.from_keras_model(model)`
- `converter.optimizations = [tf.lite.Optimize.DEFAULT]`
- `converter.target_spec.supported_types = [tf.float16]` (float16 nicemleme — tezde belirttiğin gibi)
- Çıktıyı `models/heart_sound_classifier_fp16.tflite` olarak kaydet.
- **Tutarlılık testi:** Test setinin tamamında hem Keras hem TFLite modelini çalıştır, metrikleri karşılaştır, farkı raporla. Bu tablo da Bölüm 4'e gidecek.
- Model boyutunu (MB) ve ortalama çıkarım süresini (ms) ölç ve raporla.

---

## Aşama 11 — Android Entegrasyonu için Hazırlık

- `models/heart_sound_classifier_fp16.tflite` dosyasını Android projendeki `app/src/main/assets/` dizinine kopyalanacak şekilde hazırla.
- `models/model_metadata.json`: girdi şekli, normalizasyon parametreleri, eşik değeri, sınıf etiketleri, Mel parametreleri (Android tarafında aynı ön işlemeyi yapmak için kritik).
- Android tarafında Mel-Spektrogram'ı **birebir aynı parametrelerle** üretmek zorundasın — aksi halde train/serve mismatch olur. TarsosDSP veya kendi Kotlin implementasyonunu yazmadan önce Python çıktılarıyla nümerik karşılaştırma testi yap.

---

## 📸 Tez İçin Alınması Gereken Ekran Görüntüleri ve Figürler

Aşağıdaki görselleri ilgili aşamada üretip `figures/` klasörüne kaydet. Her birinin tezde nereye gireceği de belirtildi.

### Aşama 1 sonrası — Veri Seti İstatistikleri
- **`fig_dataset_distribution.png`** — Her iki veri setindeki Normal/Anormal kayıt sayılarının bar grafiği (PhysioNet ve PASCAL ayrı ayrı + birleşik).
- *Tez konumu:* Bölüm 3.1.1, Çizelge 3.1'in altına.

### Aşama 2 sonrası — Segmentleme Sonuçları
- **`fig_segment_count.png`** — Segmentleme sonrası sınıf başına segment sayısı.
- **`fig_raw_pcg_example.png`** — Örnek bir 5 saniyelik ham PCG segmentinin zaman ekseninde dalga formu (Normal ve Anormal'dan birer örnek, yan yana).
- *Tez konumu:* Bölüm 3.1.2.

### Aşama 3 sonrası — Filtreleme Etkisi
- **`fig_filter_before_after.png`** — Aynı segmentin filtreleme öncesi ve sonrası dalga formu karşılaştırması (üst panel: ham, alt panel: 25–900 Hz Butterworth sonrası).
- **`fig_frequency_response.png`** — Tasarlanan Butterworth filtresinin frekans yanıtı (Bode plot).
- *Tez konumu:* Bölüm 3.1.3.

### Aşama 4 sonrası — Mel-Spektrogram Örnekleri ⭐
- **`fig_mel_normal.png`** — Normal bir kayda ait Mel-Spektrogram (224×224, dB ölçekli, colorbar'lı).
- **`fig_mel_abnormal.png`** — Anormal bir kayda ait Mel-Spektrogram.
- **`fig_mel_comparison.png`** — Yukarıdaki ikisinin yan yana karşılaştırması (tek figürde subplot).
- *Tez konumu:* Bölüm 3.1.4, Çizelge 3.2'nin altına.

### Aşama 5 sonrası — Veri Artırma Görselleri
- **`fig_augmentation_examples.png`** — Aynı Mel-Spektrogramın 5 farklı versiyonu yan yana: (1) orijinal, (2) zaman kaydırılmış, (3) zaman maskelenmiş, (4) frekans maskelenmiş, (5) genlik ölçeklenmiş.
- *Tez konumu:* Bölüm 3.1.4 sonu (veri artırma anlatımının altı).

### Aşama 6 sonrası — Veri Bölümleme
- **`fig_split_distribution.png`** — Eğitim/Doğrulama/Test setlerindeki Normal/Anormal dağılımının yığılmış bar grafiği.
- *Tez konumu:* Bölüm 3.4.1, "Veri seti bölümleme" paragrafının altı.

### Aşama 7 sonrası — Model Mimarisi
- **`fig_model_architecture.png`** — `tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)` çıktısı. Veya MobileNetV2 + sınıflandırma başlığını gösteren elle çizilmiş bir blok diyagram (draw.io / diagrams.net önerilir).
- **Ek olarak:** `model.summary()` çıktısının ekran görüntüsü → `fig_model_summary.png`.
- *Tez konumu:* Bölüm 3.4.1.

### Aşama 8 sonrası — Eğitim Eğrileri ⭐ (Tezdeki "Şekil 3.X")
- **`fig_training_curves.png`** — Tezde doğrudan adı geçen şekil. 2×1 subplot:
  - Üst panel: Eğitim ve doğrulama **kaybı** (loss) — epoch ekseninde, iki çizgi.
  - Alt panel: Eğitim ve doğrulama **doğruluğu** (accuracy) — epoch ekseninde, iki çizgi.
  - Aşama 1 ve Aşama 2 arasındaki geçişi dikey kesik çizgiyle işaretle ve "Fine-tuning başlangıcı" etiketi ekle.
- **`fig_lr_schedule.png`** — ReduceLROnPlateau'nun epoch boyunca öğrenme oranını nasıl düşürdüğünün grafiği (CSVLogger çıktısından üretilebilir).
- **Colab ekran görüntüleri:**
  - `screenshot_colab_gpu.png` — Runtime → Change runtime type → T4 GPU seçili ekran (donanım kanıtı için).
  - `screenshot_colab_training.png` — Eğitim sırasında ilerleyen epoch çıktıları (`model.fit` log'u).
- *Tez konumu:* Bölüm 3.4.2, **"Şekil 3.X"** olarak referans verilen yer burası.

### Aşama 9 sonrası — Test Seti Performansı ⭐
- **`fig_confusion_matrix.png`** — Test seti üzerindeki karışıklık matrisi (seaborn heatmap, sayılar + yüzdeler).
- **`fig_roc_curve.png`** — ROC eğrisi + AUC değeri lejantta.
- **`fig_pr_curve.png`** — Precision-Recall eğrisi.
- **`fig_metrics_table.png`** — Accuracy, Sensitivity, Specificity, Precision, F1, AUC değerlerinin tablo görseli (veya doğrudan markdown tablo olarak Bölüm 4'e yerleştir).
- *Tez konumu:* Bölüm 4 (Bulgular).

### Aşama 10 sonrası — TFLite Karşılaştırma
- **`fig_keras_vs_tflite.png`** — Keras ve TFLite (float16) modellerinin test seti metriklerinin yan yana bar grafiği.
- **`fig_model_size_inference.png`** — Model boyutu (MB) ve ortalama çıkarım süresi (ms) karşılaştırması.
- *Tez konumu:* Bölüm 3.4.2 sonu + Bölüm 4.

---

### Görsel Üretim İpuçları

- **Tutarlı stil:** Tüm figürlerde aynı font (örn. DejaVu Sans), aynı renk paleti (`seaborn.set_palette('colorblind')`), `dpi=300` ile kaydet.
- **Format:** Tez yazımında vektörel `.pdf` veya `.svg` tercih et; rapor metni içinde `.png` (300 dpi) yeterli.
- **Türkçeleştirme:** Eksen etiketlerini Türkçe yaz ("Dönem (epoch)", "Kayıp", "Doğruluk", "Gerçek Sınıf", "Tahmin Edilen Sınıf").
- **Otomatikleştir:** `src/visualization/generate_figures.py` adında tek bir betik yaz — eğitim sonrası bu betiği çalıştırınca tüm figürler tek seferde üretilsin. Tezi her güncellediğinde manuel iş kalmasın.

---

## Claude Code'a Verirken Pratik Öneriler

Her aşamayı **ayrı bir oturum** olarak ver, sonunda küçük bir sanity-check (örn. "ilk 5 segmentin shape ve label'ını yazdır") iste. Aşama 4'ten itibaren her adımda önce **küçük bir alt küme (50 kayıt)** üzerinde çalıştır, pipeline doğrulandıktan sonra tam veriye geç — yoksa hatalar geç fark edilir.

Ayrıca her betiğin başına şunu eklet: `configs/config.yaml`'dan parametre okuma + `--dry-run` flag'i. Bu, tezde tekrarlanabilirlik açısından da değerli olacak.
