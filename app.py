# app.py
# Streamlit TF-IDF & Cosine Similarity App
# Dependencies: streamlit, pandas, scikit-learn, numpy

import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="TF-IDF & Cosine Similarity (Streamlit)",
    page_icon="🧮",
    layout="wide",
)

st.title("🧮 TF‑IDF & Cosine Similarity App")
st.write(
    "Aplikasi untuk menghitung **TF‑IDF** dan **Cosine Similarity** pada korpus. "
    "Unggah CSV atau tempel teks (satu dokumen per baris)."
)

# -------------- Sidebar Config --------------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    text_col_name = st.text_input("Nama kolom teks (CSV)", value="text")
    ngram_min = st.number_input("n-gram min", min_value=1, max_value=5, value=1)
    ngram_max = st.number_input("n-gram max", min_value=1, max_value=5, value=1)
    if ngram_max < ngram_min:
        ngram_max = ngram_min
    max_features = st.number_input("Max features (0=auto)", min_value=0, max_value=200000, value=0, step=1000)
    use_idf = st.checkbox("Gunakan IDF (TF‑IDF). Jika tidak, hanya TF.", True)
    norm_opt = st.selectbox("Normalisasi vektor", ["l2", "l1", None], index=0)
    use_english_stop = st.checkbox("Stopwords English", False)
    lowercase = st.checkbox("Lowercase", True)
    binary = st.checkbox("Binary counts (TF)", False)

st.markdown("### 1) Masukkan Data")
tab_upload, tab_paste, tab_sample = st.tabs(["📤 Upload CSV", "📝 Tempel Teks", "📦 Contoh"])

docs, doc_ids = [], []

with tab_upload:
    up = st.file_uploader("Unggah CSV (harus ada kolom sesuai nama di sidebar, default: `text`)", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception:
            up.seek(0)
            df = pd.read_csv(io.BytesIO(up.read()), encoding_errors="ignore")
        if text_col_name not in df.columns:
            st.error(f"Kolom '{text_col_name}' tidak ditemukan. Kolom tersedia: {list(df.columns)}")
        else:
            docs = df[text_col_name].astype("string").fillna("").tolist()
            doc_ids = df.index.astype(str).tolist()
            st.success(f"Memuat {len(docs)} dokumen dari CSV.")
            st.dataframe(df.head(10), use_container_width=True)

with tab_paste:
    txt_area = st.text_area("Tempel teks (satu dokumen per baris)", height=160, placeholder="Dokumen 1... Dokumen 2...")
    if txt_area.strip():
        docs = [line.strip() for line in txt_area.splitlines() if line.strip()]
        doc_ids = [f"doc_{i+1}" for i in range(len(docs))]
        st.success(f"Memuat {len(docs)} dokumen dari input tempel.")

with tab_sample:
    sample_docs = [
        "Saya suka belajar NLP dan text mining untuk analisis dokumen.",
        "Pembelajaran mesin dan pemrosesan bahasa alami berkembang pesat.",
        "Natural language processing enables powerful search and chat systems.",
        "Cosine similarity compares the angle between TF-IDF vectors.",
        "TF-IDF memberikan bobot tinggi pada kata yang jarang tetapi informatif.",
        "RAG systems combine retrieval with generation for better grounding.",
    ]
    st.code("\n".join(sample_docs))
    if st.button("Gunakan Contoh Dataset"):
        docs = sample_docs
        doc_ids = [f"doc_{i+1}" for i in range(len(docs))]
        st.success("Contoh dataset dimuat.")

if not docs:
    st.info("Unggah CSV atau tempel teks untuk mulai.")
    st.stop()

# -------------- Vectorizer --------------
kwargs = dict(
    ngram_range=(int(ngram_min), int(ngram_max)),
    lowercase=lowercase,
    use_idf=use_idf,
    norm=norm_opt,
    binary=binary,
)
if max_features > 0:
    kwargs["max_features"] = int(max_features)
if use_english_stop:
    kwargs["stop_words"] = "english"

vectorizer = TfidfVectorizer(**kwargs)
X = vectorizer.fit_transform(docs)
feat_names = np.array(vectorizer.get_feature_names_out())
st.success(f"Matrix: {X.shape[0]} dokumen × {X.shape[1]} fitur")

# -------------- Top terms per document --------------
st.markdown("### 2) Top Terms per Dokumen")
top_n = st.slider("Top‑N terms", min_value=5, max_value=50, value=15, step=5)

def top_terms_for_doc(X_row, feat_names, n=15):
    row = X_row.toarray().ravel()
    if row.sum() == 0:
        return []
    idx = np.argsort(row)[::-1][:n]
    return [(feat_names[i], float(row[i])) for i in idx if row[i] > 0]

rows = []
for i in range(X.shape[0]):
    pairs = top_terms_for_doc(X[i], feat_names, top_n)
    for term, score in pairs:
        rows.append({"doc_id": doc_ids[i] if doc_ids else i, "term": term, "score": score})

if rows:
    df_top = pd.DataFrame(rows)
    st.dataframe(df_top, use_container_width=True, height=320)
    st.download_button(
        "⬇️ Unduh Top Terms (CSV)",
        data=df_top.to_csv(index=False).encode("utf-8"),
        file_name="top_terms.csv",
        mime="text/csv",
    )
else:
    st.info("Tidak ada term non‑zero. Coba atur parameter.")

# -------------- Cosine similarity --------------
st.markdown("### 3) Cosine Similarity antar Dokumen")
sim_mat = cosine_similarity(X)
df_sim = pd.DataFrame(sim_mat, index=doc_ids or range(len(docs)), columns=doc_ids or range(len(docs)))
st.dataframe(df_sim, use_container_width=True, height=360)
st.download_button(
    "⬇️ Unduh Cosine Similarity (CSV)",
    data=df_sim.to_csv().encode("utf-8"),
    file_name="cosine_similarity.csv",
    mime="text/csv",
)

# -------------- Query mode --------------
st.markdown("### 4) Query ke Korpus (opsional)")
q = st.text_input("Masukkan kueri untuk dibandingkan dengan seluruh dokumen:")
if q.strip():
    q_vec = vectorizer.transform([q])
    q_sim = cosine_similarity(q_vec, X).ravel()
    df_q = pd.DataFrame({"doc_id": doc_ids or range(len(docs)), "similarity": q_sim}).sort_values("similarity", ascending=False)
    st.dataframe(df_q, use_container_width=True, height=280)
    st.download_button(
        "⬇️ Unduh Hasil Query (CSV)",
        data=df_q.to_csv(index=False).encode("utf-8"),
        file_name="query_results.csv",
        mime="text/csv",
    )

st.caption("Catatan: Aktifkan/Nonaktifkan IDF untuk melihat perbedaan TF vs TF‑IDF. Atur n‑gram untuk frasa.")
