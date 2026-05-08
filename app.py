import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi Halaman
st.set_page_config(page_title="Retail Recommender", layout="wide")

@st.cache_data
def load_and_clean_data(file_path):
    # Membaca data Excel
    df = pd.read_excel(file_path)
    
    # 1. Bersihkan data dasar
    df = df.dropna(subset=['Description'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df['Description'] = df['Description'].str.strip().str.upper()
    
    # 2. KRUSIAL: Ambil Produk Unik berdasarkan StockCode
    # Ini mengurangi data dari 540rb baris menjadi ~3.800 baris saja.
    # Tanpa ini, Streamlit Cloud akan kehabisan RAM (Crash).
    df_unique = df.drop_duplicates(subset=['StockCode']).copy()
    
    # Filter deskripsi yang terlalu pendek
    df_unique = df_unique[df_unique['Description'].str.split().str.len() >= 2]
    
    return df_unique.reset_index(drop=True)

@st.cache_resource
def compute_similarity(descriptions):
    # Menggunakan TF-IDF untuk mengubah deskripsi menjadi angka
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(descriptions)
    
    # Hitung kemiripan dan ubah ke float16 untuk menghemat memori
    sim_matrix = cosine_similarity(tfidf_matrix).astype(np.float16)
    return sim_matrix

# --- Tampilan Utama ---
st.title("🛒 Online Retail Product Recommender")
st.markdown("Cari produk untuk mendapatkan rekomendasi item serupa berdasarkan deskripsi.")

try:
    with st.spinner('Menyiapkan data (ini hanya dilakukan sekali)...'):
        # Pastikan file Online Retail.xlsx ada di GitHub
        df_products = load_and_clean_data('Online Retail.xlsx')
        cosine_sim = compute_similarity(df_products['Description'])

    # Input Pencarian
    search_query = st.text_input("Cari nama produk (contoh: BAG, HEART, BOTTLE):", "").upper()

    # Filter pilihan berdasarkan pencarian
    if search_query:
        options_df = df_products[df_products['Description'].str.contains(search_query)]
    else:
        options_df = df_products

    if not options_df.empty:
        # Pilihan Produk
        selection_list = options_df.apply(lambda x: f"{x['StockCode']} - {x['Description']}", axis=1).tolist()
        selected_item = st.selectbox("Pilih produk spesifik:", selection_list)
        
        # Ambil index produk yang dipilih
        selected_stock_code = selected_item.split(" - ")[0]
        idx = df_products.index[df_products['StockCode'] == selected_stock_code].tolist()[0]

        # Tampilkan Produk Terpilih
        st.success(f"Menampilkan rekomendasi untuk: **{df_products.iloc[idx]['Description']}**")

        # Hitung Top 5 Rekomendasi
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6] # Ambil 5 teratas selain dirinya sendiri

        # Tampilkan Hasil
        st.subheader("Produk Serupa yang Direkomendasikan:")
        cols = st.columns(5)
        
        for i, (index, score) in enumerate(sim_scores):
            item = df_products.iloc[index]
            with cols[i]:
                st.info(f"**{item['Description']}**")
                st.write(f"Kode: {item['StockCode']}")
                st.write(f"Kemiripan: {float(score)*100:.1f}%")
    else:
        st.warning("Produk tidak ditemukan, coba kata kunci lain.")

except FileNotFoundError:
    st.error("Error: File 'Online Retail.xlsx' tidak ditemukan di repositori GitHub.")
except Exception as e:
    st.error(f"Terjadi kesalahan teknis: {e}")
