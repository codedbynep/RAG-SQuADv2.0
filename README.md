# Peningkatan Kualitas Jawaban dan Abstention pada QA berbasis RAG: Kombinasi Zero-shot Instruction Prompting dan Self-verification

Repositori ini berisi kode, dataset, dan hasil eksperimen dari penelitian skripsi yang menerapkan **Retrieval-Augmented Generation (RAG)** domain **Question Answering (QA)** pada dataset **SQuAD v2.0**.

---

## Informasi Skripsi

|                    |                                         |
|--------------------|-----------------------------------------|
| Tahun              | 2026                                    |
| Penulis            | Nevitya Elmaira Nurjannah               |
| Institusi          | Universitas Negeri Surabaya             |
| Dosen Pembimbing   | Cendra Devayana Putra, S.Kom., M.I.M.   |
| Dosen Penguji 1    | Dr. Wiyli Yustanti, S.Si., M.Kom.       |
| Dosen Penguji 2    | Monica Cinthya, M.Kom.                  |



## Ringkasan Penelitian

Penelitian ini menggunakan **desain faktorial 4×3**, yaitu 4 konfigurasi pipeline diuji pada 3 skala model dari keluarga yang sama (Qwen3).

**Empat konfigurasi pipeline:**

| Konfigurasi | Komponen |
|-------------|----------|
| Konfigurasi 1 | RAG dasar (*baseline*) |
| Konfigurasi 2 | RAG + Zero-shot Instruction Prompting |
| Konfigurasi 3 | RAG + Self-verification |
| Konfigurasi 4 | RAG + Prompting + Self-verification |

**Tiga skala model (Qwen3, same-family):**

| Skala | Model | Ukuran |
|-------|-------|--------|
| SLM | Qwen3 | 0.6B |
| MLM | Qwen3 | 4B |
| LLM | Qwen3 | 8B |

---

## Rumusan Masalah

1. Bagaimana perbandingan kualitas jawaban berbasis bukti (*evidence-based*) pada pertanyaan *answerable* di antara 4 konfigurasi?
2. Bagaimana perbandingan ketepatan *abstention* pada pertanyaan *unanswerable* di antara 4 konfigurasi?

---

## Metode Evaluasi

- **Faithfulness** diukur menggunakan pendekatan berbasis **NLI** (Natural Language Inference) dengan model `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, mengikuti kerangka RAGAS (Es et al., EACL 2024) dan divalidasi terhadap SummaC (Laban et al., TACL 2022) serta AlignScore (Zha et al., ACL 2023).
- **Abstention** diukur menggunakan metrik **Abstention F1**, dengan deteksi abstain berbasis zero-shot NLI.
- **Evaluasi manusia** dilakukan oleh tiga evaluator dengan metrik **Win Rate** dan reliabilitas antar-penilai **Fleiss' Kappa**.

## Uji Statistik

- **Faithfulness**: ART RM-ANOVA (non-parametrik, karena asumsi normalitas Shapiro-Wilk tidak terpenuhi).
- **Abstention**: Cochran's Q test.

---

## Hasil Utama

**Faithfulness (RQ1):** Konfigurasi, skala model, dan interaksinya berpengaruh signifikan (F=17.05; F=30.20; F=9.38; semua p<0.001). Penambahan self-verification cenderung menurunkan faithfulness, terutama pada SLM, sedangkan LLM tetap stabil tinggi.

**Abstention (RQ2):** Cochran's Q tidak menunjukkan perbedaan signifikan antar konfigurasi pada ketiga skala model (semua p>0.05). Terdapat kecenderungan kenaikan skor secara deskriptif, namun tidak terbukti signifikan secara statistik.

---

## Struktur Repositori

```
.
├── dataset/      Data SQuAD v2.0 yang digunakan dalam eksperimen
├── source/       Notebook eksperimen (top-k, pipeline, uji statistik)
└── results/      Output: human eval, metrics, predictions
```

---

## Cara Menjalankan

1. Clone repositori ini:
   ```bash
   git clone https://github.com/codedbynep/RAG-SQuADv2.0.git
   ```
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Buka notebook di folder `source/` secara berurutan (disarankan dijalankan di Google Colab dengan GPU).

> **Catatan:** Eksperimen dijalankan pada Google Colab (Tesla T4). Beberapa model memerlukan kuantisasi agar muat dalam VRAM.

---

## Dataset

Dataset yang digunakan adalah **SQuAD v2.0** (Rajpurkar et al., 2018), yang memuat pertanyaan *answerable* dan *unanswerable*. Subset yang digunakan dalam penelitian ini tersedia di folder `dataset/`.

---

## Teknologi

- Python (Jupyter Notebook)
- Hugging Face Transformers
- Qwen3 (0.6B / 4B / 8B)
- pingouin, scipy (uji statistik)
- Google Colab (Tesla T4)

---

## Penulis

Nevitya Elmaira Nurjannah — Program Studi Sistem Informasi, Universitas Negeri Surabaya (UNESA)

Penelitian ini merupakan bagian dari skripsi tahun 2026.
