import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi Halaman
st.set_page_config(page_title="Retail Product Recommender", layout="wide")

# --- STYLE ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD & PREPROCESSING ---
@st.cache_data
def load_and_clean_data(file_path):
    # Load data
    df = pd.read_excel(file_path)
    
    # Cleaning
    df_clean = df.dropna(subset=['Description']).copy()
    df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
    
    # Create Product Catalog
    df_products = df_clean.groupby('StockCode').agg(
        Description=('Description', lambda x: x.mode()[0]),
        TotalSold=('Quantity', 'sum'),
        UnitPrice=('UnitPrice', 'mean')
    ).reset_index()
    
    df_products['Description'] = df_products['Description'].str.strip().str.upper()
    # Filter produk dengan deskripsi minimal 3 kata
    df_products = df_products[df_products['Description'].str.split().str.len() >= 3]
    return df_products.reset_index(drop=True)

@st.cache_resource
def compute_similarity(descriptions):
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(descriptions)
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return sim_matrix

# --- SIDEBAR ---
st.sidebar.title("⚙️ Konfigurasi")
top_n = st.sidebar.slider("Jumlah Rekomendasi", 1, 10, 5)

# --- MAIN APP ---
st.title("🛒 Product Recommendation System")
st.markdown("Sistem rekomendasi berbasis **konten (deskripsi)** menggunakan TF-IDF & Cosine Similarity.")

try:
    # Load Data
    with st.spinner('Memproses data retail...'):
        df_products = load_and_clean_data('Online Retail.xlsx')
        cosine_sim = compute_similarity(df_products['Description'])

    # --- PENCARIAN ---
    st.subheader("🔍 Cari & Pilih Produk")
    search_query = st.text_input("Masukkan kata kunci produk (contoh: BAG, HEART, CANDLE)", "")

    # Filter produk berdasarkan search
    filtered_products = df_products[df_products['Description'].str.contains(search_query.upper())] if search_query else df_products

    if not filtered_products.empty:
        # User memilih produk dari hasil pencarian
        product_options = filtered_products.apply(lambda x: f"{x['StockCode']} - {x['Description']}", axis=1).tolist()
        selected_option = st.selectbox("Pilih produk spesifik untuk melihat rekomendasi:", product_options)
        
        selected_stock_code = selected_option.split(" - ")[0]
        
        # Ambil detail produk yang dipilih
        idx = df_products.index[df_products['StockCode'] == selected_stock_code].tolist()[0]
        selected_item = df_products.loc[idx]

        # Tampilkan Detail Produk Terpilih
        st.info(f"**Produk Terpilih:** {selected_item['Description']}")
        col1, col2, col3 = st.columns(3)
        col1.metric("StockCode", selected_item['StockCode'])
        col2.metric("Harga Rata-rata", f"£{selected_item['UnitPrice']:.2f}")
        col3.metric("Total Terjual", f"{int(selected_item['TotalSold'])} unit")

        st.divider()

        # --- HITUNG REKOMENDASI ---
        st.subheader(f"✨ {top_n} Produk Serupa yang Mungkin Anda Sukai")
        
        # Ambil skor kemiripan
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1] # Ambil top N, skip produk itu sendiri

        product_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]

        # Tampilkan Hasil dalam Tabel
        results = df_products.iloc[product_indices][['StockCode', 'Description', 'UnitPrice', 'TotalSold']].copy()
        results['Similarity Score'] = [f"{round(s*100, 2)}%" for s in similarity_scores]
        
        st.table(results.reset_index(drop=True))

    else:
        st.warning("Produk tidak ditemukan. Coba kata kunci lain.")

except FileNotFoundError:
    st.error("File 'Online Retail.xlsx' tidak ditemukan! Pastikan file berada di folder yang sama.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")

# Footer
st.caption("Data: Online Retail Dataset | Metode: Content-Based Filtering")