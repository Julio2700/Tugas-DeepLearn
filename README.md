# Deep Learning: CNN & Transfer Learning pada CIFAR-10

Tugas individu mata kuliah Deep Learning. Repositori ini mengimplementasikan **2 arsitektur**:

1. **CNN (dari nol)**: 3 blok Conv-BatchNorm-MaxPool-Dropout + Global Average Pooling.
2. **Transfer Learning**: MobileNetV2 pretrained ImageNet (feature extraction, lalu fine-tuning).

## Dataset
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): 60.000 gambar berwarna 32x32 dalam 10 kelas
(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
Pembagian data: 45.000 train, 5.000 validasi, 10.000 test.

## Arsitektur
| Model | Ringkasan | Strategi training |
|-------|-----------|-------------------|
| CNN | 6 Conv2D (32-64-128) + BatchNorm + MaxPool + Dropout, GAP, Dense 128 | Adam 1e-3, EarlyStopping, ReduceLROnPlateau |
| MobileNetV2 | Input di-resize ke 96x96, base dibekukan, head GAP-Dropout-Dense | Phase 1: base beku (lr 1e-3). Phase 2: fine-tune 30 layer terakhir (lr 1e-5) |

## Hasil
> Isi tabel ini setelah menjalankan notebook/script.

| Model | Test Accuracy | Test Loss |
|-------|---------------|-----------|
| CNN | ... | ... |
| MobileNetV2 (Transfer Learning) | ... | ... |

Grafik training dan confusion matrix tersimpan di folder `results/`.

## Analisis singkat
> Tulis analisis dengan kata-katamu sendiri: model mana yang lebih baik, kenapa,
> kelas mana yang sering tertukar (misalnya cat vs dog), ada overfitting atau tidak,
> dan perbandingan waktu training.

## Cara menjalankan
**Google Colab (disarankan):** upload `notebook.ipynb`, aktifkan GPU (Runtime > Change runtime type > T4 GPU), lalu Run all.

**Lokal:**
```bash
pip install -r requirements.txt
python train.py
```

## Struktur
```
.
├── README.md
├── requirements.txt
├── notebook.ipynb      # eksperimen lengkap + visualisasi
├── train.py            # versi script
├── src/
│   ├── common.py       # load data, plot, evaluasi
│   └── models.py       # definisi CNN & transfer learning
└── results/            # grafik & ringkasan hasil
```
