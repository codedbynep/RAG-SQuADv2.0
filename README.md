# Peningkatan Kualitas Jawaban dan Abstention pada QA berbasis RAG: Kombinasi Zero-shot Instruction Prompting dan Self-verification

Repositori ini merupakan kode pendukung skripsi dengan judul **"Peningkatan Kualitas Jawaban dan Abstention pada QA berbasis RAG: Kombinasi Zero-shot Instruction Prompting dan Self-verification"**.

## Informasi Skripsi

| | |
|---|---|
| Tahun | 2026 |
| Penulis | Nevitya Elmaira Nurjannah |
| Afiliasi | Universitas Negeri Surabaya |
| Dosen Pembimbing | Cendra Devayana Putra, S.Kom., M.I.M. |
| Dosen Penguji 1 | Dr. Wiyli Yustanti, S.Si., M.Kom. |
| Dosen Penguji 2 | Monica Cinthya, M.Kom. |

## Deskripsi Singkat

Proyek ini membangun sistem **question answering (QA) berbasis bukti** menggunakan pendekatan **Retrieval-Augmented Generation (RAG)**. Sistem menerima sebuah pertanyaan, mengambil potongan konteks (*passage*) paling relevan dari basis pengetahuan, lalu menghasilkan jawaban yang harus bersandar pada konteks tersebut, atau memilih **abstain** (menolak menjawab) ketika bukti tidak tersedia.

Dataset yang digunakan adalah **SQuAD v2.0** yang memuat pertanyaan *answerable* dan *unanswerable*. Selain menguji pipeline RAG dasar, penelitian ini menganalisis pengaruh dua teknik penguat, yaitu **zero-shot instruction prompting** dan **self-verification**, terhadap performa sistem melalui empat konfigurasi yang diuji pada tiga skala model.

## Pipeline / Alur Sistem

1. **Persiapan & pra-pemrosesan data** — memuat dataset SQuAD v2.0, melakukan pembersihan teks (*replace newline/tab*, normalisasi spasi, *trim*), lalu menyaring konteks unik melalui *deduplication*.
2. **Embedding** — merepresentasikan setiap konteks unik sebagai vektor menggunakan model `Qwen3-Embedding` (0.6B / 4B / 8B) dengan `normalize_embeddings=True`.
3. **Indexing** — mengindeks seluruh vektor konteks dengan **FAISS** (`IndexFlatIP`, setara *cosine similarity* karena vektor telah dinormalisasi).
4. **Retrieval** — memvektorkan *query* lalu mengambil *passage* top-k paling relevan dari indeks FAISS.
5. **Augmentation** — menyusun konteks hasil *retrieval* bersama *query* menjadi *prompt* masukan model (titik penyisipan instruksi *zero-shot*).
6. **Generation** — menghasilkan jawaban dari *prompt* menggunakan model generatif `Qwen3` (0.6B / 4B / 8B).
7. **Self-verification** — tahap *verify* opsional yang membandingkan kandidat jawaban dengan konteks dan memberi label `SUPPORTED`/`UNSUPPORTED`, lalu memperbaiki jawaban atau memicu *abstain*.
8. **Evaluasi** — mengukur performa dengan **Faithfulness (NLI)** dan **Abstention F1** sebagai metrik utama, serta **EM** dan **F1-token** sebagai penunjang, dilengkapi *human evaluation*.

## Skenario Eksperimen

Penelitian menguji empat konfigurasi pipeline pada tiga skala model dari satu keluarga (Qwen3), menghasilkan desain **faktorial 4×3**.

| Konfigurasi | Zero-shot Prompting | Self-verification |
|---|---|---|
| Konfigurasi 1 (RAG dasar) | – | – |
| Konfigurasi 2 (RAG + Prompting) | ✔ | – |
| Konfigurasi 3 (RAG + Self-verification) | – | ✔ |
| Konfigurasi 4 (RAG + Prompting + Self-verification) | ✔ | ✔ |

| Skala | Model generatif | Model embedding | Ukuran |
|---|---|---|---|
| SLM | `Qwen3-0.6B` | `Qwen3-Embedding-0.6B` | 0.6B |
| MLM | `Qwen3-4B` | `Qwen3-Embedding-4B` | 4B |
| LLM | `Qwen3-8B` | `Qwen3-Embedding-8B` | 8B |

## Struktur Repositori

```
.
├── dataset/      # Subset SQuAD v2.0 yang digunakan dalam eksperimen
├── source/       # Notebook eksperimen
│   ├── topk_validation.ipynb       # Validasi pemilihan top-k retrieval
│   ├── pipeline_rag.ipynb          # Pipeline 4 konfigurasi (retrieval → generation → verify)
│   └── uji_statistik.ipynb         # ART RM-ANOVA & Cochran's Q
└── results/      # Output: prediksi, skor metrik, dan human evaluation
```

## Teknologi yang Digunakan

- **Bahasa & lingkungan**: Python, Jupyter Notebook (dikembangkan di Google Colab, Tesla T4)
- **Model generatif**: `Qwen3` — `0.6B` / `4B` / `8B`
- **Embedding**: `Qwen3-Embedding` — `0.6B` / `4B` / `8B`
- **Vector index**: `faiss-cpu` (`IndexFlatIP`)
- **Faithfulness (NLI)**: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- **Uji statistik**: `pingouin`, `statsmodels`, `scipy`
- **Pendukung**: `transformers`, `sentence-transformers`, `accelerate`, `bitsandbytes` (kuantisasi 4-bit pada LLM)

## Cara Menjalankan

1. *Clone* repositori ini, lalu buka folder `source/`.
   ```bash
   git clone https://github.com/codedbynep/RAG-SQuADv2.0.git
   ```
2. Pasang dependensi yang dibutuhkan (juga tersedia pada sel pertama tiap notebook):
   ```bash
   pip install -q transformers>=4.51.0 sentence-transformers>=2.7.0 accelerate "numpy<2" faiss-cpu bitsandbytes
   ```
3. *Restart runtime* setelah instalasi agar library yang baru dapat digunakan.
4. Jalankan notebook RAG sesuai skala model dan konfigurasi yang ingin direproduksi, lalu amati hasil evaluasinya.

> **Catatan:** Disarankan menjalankan notebook pada lingkungan dengan akselerasi **GPU**. Beberapa model memerlukan kuantisasi 4-bit agar muat dalam VRAM Tesla T4.

## Dataset

Dataset berasal dari **SQuAD v2.0** ([`rajpurkar/squad_v2`](https://huggingface.co/datasets/rajpurkar/squad_v2)) yang memuat pasangan konteks–pertanyaan beserta anotasi jawaban, dengan pertanyaan *answerable* maupun *unanswerable*. Subset yang digunakan dalam penelitian ini tersedia pada folder `dataset/`.
