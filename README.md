# TF-IDF & Cosine Similarity App (Streamlit + Pandas)

Aplikasi web sederhana untuk melakukan **feature extraction** (TF‑IDF) dan menghitung **Cosine Similarity** pada korpus/dataset.

## Fitur
- Unggah **CSV** (kolom teks dapat diatur, default `text`) atau tempel teks (satu dokumen per baris)
- Atur **n‑gram**, **max_features**, **lowercase**, **stopwords English**, **TF atau TF‑IDF**, dan normalisasi vektor
- Lihat **Top Terms per dokumen**, **Cosine Similarity antar dokumen**, dan **pencarian kueri** ke seluruh korpus
- Unduh hasil sebagai **CSV**

## Instalasi (Lokal)
Disarankan memakai virtual environment.

### Windows (PowerShell/CMD)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Menjalankan Aplikasi
```bash
streamlit run app.py
```
Buka URL yang muncul (biasanya `http://localhost:8501`).

## Cara Pakai Singkat
1. Siapkan data:
   - Upload file **CSV** dengan kolom teks (default `text`) atau
   - Tempel teks (satu dokumen per baris) pada tab **Tempel Teks**
2. Atur parameter di **sidebar** (n-gram, stopwords, IDF, dst.)
3. Lihat **Top Terms**, **Cosine Similarity**, serta **Query ke Korpus**
4. Unduh tabel hasil dalam format **CSV**

## Contoh Data
Gunakan `sample_data.csv` untuk uji cepat.
