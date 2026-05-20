import streamlit as st
import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz
import jellyfish 
# >>> IMPORT BARU UNTUK AI & VECTOR DATABASE <<<
from sentence_transformers import SentenceTransformer
import faiss 

st.set_page_config(page_title="AI-Powered Movie Search", layout="wide")

# ==========================================
# 1. AI & VECTOR DATABASE INITIALIZATION
# ==========================================
@st.cache_resource
def load_ai_model():
    """Memuat model AI NLP sekali saat startup. Model ini memahami bahasa manusia."""
    # Model 'all-MiniLM-L6-v2' sangat cepat dan akurat untuk kesamaan kalimat
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def build_vector_database(df_clean):
    """Membangun Vector Database (FAISS) dari judul film yang sudah bersih."""
    # Ambil judul film
    titles = df_clean['title'].tolist()
    
    # Ubah teks menjadi Vektor (Array multidimensi yang merepresentasikan makna kata)
    with st.spinner("🧠 Membangun Database AI... (Ini membutuhkan waktu beberapa detik)"):
        embeddings = model.encode(titles, show_progress_bar=False)
    
    # Normalisasi vektor ( agar perhitungan jarak lebih akurat)
    faiss.normalize_L2(embeddings)
    
    # Buat Index FAISS (Vector Database di RAM)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Menggunakan Inner Product (Cosine Similarity)
    index.add(embeddings)
    
    return index

model = load_ai_model()

# ==========================================
# 2. HELPER FUNCTIONS (ALGORITMA PENCARIAN HYBRID)
# ==========================================

@st.cache_data
def preprocess_data(df):
    df['search_key'] = df['title'].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)
    df['meta_key'] = df['search_key'].apply(lambda x: ' '.join([jellyfish.metaphone(w) for w in str(x).split()]))
    df['sorted_key'] = df['search_key'].apply(lambda x: ' '.join(sorted(x.split())))
    return df

def normalize_query(text):
    if not text: return ""
    text = str(text).lower().strip()
    num_map = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "satu": "1", "dua": "2", "tiga": "3", "empat": "4", "lima": "5"}
    words = text.split()
    normalized_words = [num_map.get(w, w) for w in words]
    text = ' '.join(normalized_words)
    roman_map = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6"}
    roman_pattern = re.compile(r'\b([ivx]+)\b$', re.IGNORECASE)
    match = roman_pattern.search(text)
    if match and match.group(1).lower() in roman_map:
        text = text[:match.start()].strip() + " " + roman_map[match.group(1).lower()]
    return text

def get_ai_suggestions(query, vector_index, df_search, top_k=5):
    """Mencari kemiripan makna menggunakan AI Vector Database"""
    # Ubah query user menjadi vektor
    query_vector = model.encode([query])
    faiss.normalize_L2(query_vector)
    
    # Cari di FAISS database
    distances, indices = vector_index.search(query_vector, top_k)
    
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        score = distances[0][i] # Skor kemiripan (0-1, semakin mendekati 1 semakin mirip)
        if idx != -1 and score > 0.4: # Threshold AI
            results.append({
                'index': idx,
                'title': df_search.iloc[idx]['title'],
                'score': score * 100 # Ubah ke persentase
            })
    return results

def smart_search_ultimate(query, vector_index, df_search, top_n=5):
    """
    Algoritma Hybrid Terakhir:
    1. Coba Exact/Fuzzy dulu (super cepat untuk typo kecil)
    2. Jika tidak ketemu / kurang meyakinkan, gunakan AI Vector Search (untuk typo ekstrem & bahasa manusia)
    """
    if not query or len(query) < 2:
        return pd.DataFrame()
    
    q_norm = normalize_query(query)
    
    # LANGKAH 1: CEK EXACT MATCH (Instan)
    exact_idx = next((i for i, k in enumerate(df_search['search_key']) if q_norm == k), None)
    if exact_idx is not None:
        return df_search.iloc[[exact_idx]]
    
    # LANGKAH 2: FUZZY & PHONETIC MATCH (Untuk typo huruf)
    q_meta = ' '.join([jellyfish.metaphone(w) for w in q_norm.split()])
    q_sorted = ' '.join(sorted(q_norm.split()))
    
    best_fuzz_score = 0
    best_fuzz_idx = -1
    
    for i, row in df_search.iterrows():
        score_fuzz = max(fuzz.partial_ratio(q_norm, row['search_key']), fuzz.token_sort_ratio(q_norm, row['search_key']))
        score_phonetic = fuzz.token_set_ratio(q_meta, row['meta_key'])
        final_hybrid = (score_fuzz * 0.7) + (score_phonetic * 0.3)
        
        if final_hybrid > best_fuzz_score:
            best_fuzz_score = final_hybrid
            best_fuzz_idx = i
            
    # Jika Fuzzy menemukan skor sangat tinggi (>85%), Langsung kembalikan hasilnya
    if best_fuzz_score > 85:
        return df_search.iloc[[best_fuzz_idx]]

    # LANGKAH 3: AI VECTOR SEARCH (Fallback untuk kasus ekstrem / bahasa manusia)
    # Jika Fuzzy menemukan sesuatu tapi rendah (<85%), gabungkan dengan AI
    ai_results = get_ai_suggestions(query, vector_index, df_search, top_k=top_n)
    
    # Jika AI menemukan sesuatu yang lebih meyakinkan
    if ai_results:
        best_ai_score = ai_results[0]['score']
        
        # Bandingkan pemenang Fuzzy vs pemenang AI
        if best_ai_score > best_fuzz_score:
            # AI menang, kembalikan hasil AI
            final_indices = [r['index'] for r in ai_results]
            return df_search.iloc[final_indices]
        else:
            # Fuzzy menang
            return df_search.iloc[[best_fuzz_idx]]
            
    # Terakhir, kembalikan hasil fuzzy terbaik walaupun skor rendah (sebagai last resort)
    if best_fuzz_idx != -1 and best_fuzz_score > 50:
         return df_search.iloc[[best_fuzz_idx]]
         
    return pd.DataFrame()

# ==========================================
# 3. MAIN APP LOGIC
# ==========================================
@st.cache_data
def load_and_process_data(file):
    df_raw = pd.read_csv(file)
    df_raw.columns = df_raw.columns.str.strip()
    
    df_rejected_list = []
    # Proses Rejection sama seperti sebelumnya (singkat saya tulis ulang)
    mask_dup = df_raw.duplicated(subset=['title'], keep='first')
    temp_date = pd.to_datetime(df_raw['release_date'], errors='coerce')
    mask_date = temp_date.isna()
    mask_lang = df_raw['original_language'].isna()
    mask_overview = df_raw['overview'].str.len() < 10

    masks = [mask_dup, mask_date, mask_lang, mask_overview]
    reasons = ['Duplikat Judul', 'Tanggal Rilis Kosong/Rusak', 'Bahasa Asli Kosong', 'Deskripsi Tidak Memadai']
    
    for mask, reason in zip(masks, reasons):
        for idx, row in df_raw[mask].iterrows():
            if idx not in [r['original_index'] for r in df_rejected_list]:
                df_rejected_list.append({'original_index': idx, 'reason': reason, **row.to_dict()})

    indices_to_drop = [r['original_index'] for r in df_rejected_list] if df_rejected_list else []
    df_rejected = pd.DataFrame(df_rejected_list) if df_rejected_list else pd.DataFrame()
    
    df_clean = df_raw.drop(index=indices_to_drop).copy()
    df_clean = df_clean.sort_values(by='title').reset_index(drop=True)
    df_clean = preprocess_data(df_clean)
    
    return df_raw, df_rejected, df_clean

st.title("🧠 AI-Powered Movie Data Engineering & Search")
st.caption("Menggunakan Hybrid Algorithm: Fuzzy String + Phonetic + Vector AI Database")

uploaded_file = st.file_uploader("Unggah Dataset TMDB", type=["csv"])

if uploaded_file is not None:
    with st.spinner("Memproses data mentah..."):
        df_raw, df_rejected, df_clean = load_and_process_data(uploaded_file)
    
    # BUAT VECTOR DATABASE SAAT DATA BERSIH DIBUAT
    vector_db = build_vector_database(df_clean)

    # VISUALISASI (Sama seperti sebelumnya)
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("📦 Total Data Raw", f"{len(df_raw)}")
    c2.metric("🗑️ Data Dibuang", f"{len(df_rejected)}", delta=f"-{len(df_rejected)}", delta_color="inverse")
    c3.metric("✅ Data Bersih + AI Index", f"{len(df_clean)}")

    tab_raw, tab_rejected, tab_clean, tab_search = st.tabs([
        "🔴 Data Mentah", "🟠 Data Terbuang", "🟢 Data Bersih", "🤖 AI SMART SEARCH"
    ])

    with tab_raw: st.dataframe(df_raw, use_container_width=True, height=300)
    with tab_rejected: 
        if not df_rejected.empty:
            cols_show = ['reason', 'title', 'release_date']
            st.dataframe(df_rejected[cols_show], use_container_width=True, height=300)
    with tab_clean: st.dataframe(df_clean.drop(columns=['search_key', 'meta_key', 'sorted_key']), use_container_width=True, height=300)

    with tab_search:
        st.subheader("Coba Uji Kecerdasan Buatan (AI)")
        st.markdown("""
        Test Case tingkat lanjut yang bisa dicoba:
        1. **Typo Ekstrem:** `termiet tiga` -> *Terminator 3*
        2. **Bahasa Natural:** `film tentang robot dari masa depan` -> *The Terminator*
        3. **Typo Fonetik + Natural:** `pisang kapal tenggelam` -> *Titanic* (Jika ada di database, biasanya Titanic berhasil ditemukan vector AI-nya karena konteks tenggelam)
        4. **Kesalahan Total:** `spiderman verses the spider verse` -> *Spider-Man: Into the Spider-Verse*
        """)
        
        keyword = st.text_input("Ketik apapun (Bisa typo, bisa bahasa manusia):", key="ai_search")
        
        if keyword:
            results_df = smart_search_ultimate(keyword, vector_db, df_clean, top_n=5)
            
            if not results_df.empty:
                options = results_df['title'].tolist()
                selected_title = st.selectbox("🔮 Saran AI:", options=options, index=0)
                
                if selected_title:
                    movie_data = df_clean[df_clean['title'] == selected_title].iloc[0]
                    col_info, col_overview = st.columns([1, 2])
                    with col_info:
                        st.info(f"**{movie_data['title']}**")
                        st.write(f"📅 {movie_data['release_date']}")
                        st.write(f"⭐ {movie_data['vote_average']}")
                    with col_overview:
                        st.write(movie_data['overview'])
            else:
                st.error("AI tidak dapat menemukan kemiripan makna yang cukup.")