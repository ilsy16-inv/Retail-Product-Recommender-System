import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi Halaman
st.set_page_config(page_title="Retail Recommendation System", layout="wide")

@st.cache_data
def load_and_preprocess_data(file_path):
    # Load dataset
    df_raw = pd.read_excel(file_path)
    
    # Preprocessing: Bersihkan data kosong dan transaksi retur
    df_clean = df_raw.dropna(subset=['Description']).copy()
    df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
    df_clean = df_clean[df_clean['Quantity'] > 0]
    df_clean = df_clean[df_clean['UnitPrice'] > 0]
    
    # Buat katalog produk unik berdasarkan StockCode
    df_products = df_clean.groupby('StockCode').agg(
        Description=('Description', lambda x: x.mode()[0]),
        UnitPrice=('UnitPrice', 'mean')
    ).reset_index()
    
    # Standarisasi deskripsi
    df_products['Description'] = df_products['Description'].str.strip().str.upper()
    
    # Filter produk dengan deskripsi yang cukup panjang (minimal 3 kata)
    df_products = df_products[df_products['Description'].str.split().str.len() >= 3]
    return df_products.reset_index(drop=True)

@st.cache_resource
def build_model(df):
    # Inisialisasi TF-IDF Vectorizer (unigram + bigram)
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['Description'])
    
    # Hitung Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# --- UI STREAMLIT ---
st.title("🛍️ Content-Based Product Recommender")
st.markdown("""
Sistem ini memberikan rekomendasi produk berdasarkan **kemiripan deskripsi** menggunakan algoritma **TF-IDF** dan **Cosine Similarity**.
""")

try:
    # 1. Load Data
    data_load_state = st.text('Memuat data...')
    df_products = load_and_preprocess_data('Online Retail.xlsx')
    data_load_state.empty()

    # 2. Build Model
    cosine_sim = build_model(df_products)

    # 3. Sidebar: Pilih Produk
    st.sidebar.header("Pilih Produk")
    product_list = df_products['Description'].tolist()
    selected_product = st.sidebar.selectbox("Cari atau pilih produk:", product_list)
    
    top_n = st.sidebar.slider("Jumlah rekomendasi:", 5, 20, 10)

    # 4. Logika Rekomendasi
    if selected_product:
        # Cari index produk
        idx = df_products[df_products['Description'] == selected_product].index[0]
        
        # Ambil skor kemiripan
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Ambil Top-N (abaikan produk itu sendiri di index 0)
        sim_scores = sim_scores[1:top_n+1]
        product_indices = [i[0] for i in sim_scores]
        recommendations = df_products.iloc[product_indices].copy()
        recommendations['Similarity Score'] = [i[1] for i in sim_scores]

        # 5. Tampilan Hasil
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Produk Terpilih")
            st.info(f"**{selected_product}**")
            st.write(f"StockCode: {df_products.iloc[idx]['StockCode']}")

        with col2:
            st.subheader(f"Top {top_n} Rekomendasi Produk Serupa")
            st.table(recommendations[['StockCode', 'Description', 'Similarity Score']])

except FileNotFoundError:
    st.error("File 'Online Retail.xlsx' tidak ditemukan. Pastikan file ada di folder yang sama.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")