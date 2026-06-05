# Eksperimen SML — M. Rohid Rivaldi

Repo ini berisi eksperimen awal dan script preprocessing untuk proyek akhir kelas **Membangun Sistem Machine Learning (MSML)** di Dicoding.

## 📂 Struktur Folder

```
Eksperimen_SML_M._Rohid_Rivaldi/
├── preprocessing/
│   ├── Eksperimen_M._Rohid_Rivaldi.ipynb   # Notebook EDA & preprocessing manual
│   ├── automate_M._Rohid_Rivaldi.py        # Script otomatisasi preprocessing
│   ├── requirements.txt                     # Dependensi Python
│   └── garbage_preprocessing/               # Output hasil preprocessing (generated)
├── .github/workflows/
│   └── preprocessing.yml                    # GitHub Actions — auto preprocessing
└── README.md
```

## 🗑️ Dataset

Dataset yang digunakan: [Garbage Classification (12 Classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)  
Total: 15.515 gambar, 12 kelas kategori sampah.

**Kelas**: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass

## 🔧 Cara Pakai

### Manual (di notebook)
Buka `preprocessing/Eksperimen_M._Rohid_Rivaldi.ipynb` dan jalankan cell-nya satu per satu.

### Otomatis (via script)
```bash
pip install -r preprocessing/requirements.txt
python preprocessing/automate_M._Rohid_Rivaldi.py --input ./raw_data --output ./preprocessing/garbage_preprocessing
```

### Via GitHub Actions
Push perubahan ke branch `main` di folder `preprocessing/`, atau trigger manual lewat tab Actions.

## 👤 Author
**M. Rohid Rivaldi** — rezoku (Dicoding)

# Final Submission Setup
