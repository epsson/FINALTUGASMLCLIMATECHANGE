import streamlit as st
import joblib

st.set_page_config(page_title="Sentimen Climate Change", page_icon="🌍", layout="centered")

@st.cache_resource
def load_model_dan_vectorizer():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_model_dan_vectorizer()
except Exception as e:
    st.error("Error: File model.pkl atau vectorizer.pkl tidak ditemukan!")
    st.stop()

st.title("🌍 Analisis Sentimen Publik: Perubahan Iklim")
st.markdown("Aplikasi Machine Learning berbasis **Naive Bayes** untuk mendeteksi sentimen cuitan Twitter mengenai *Climate Change*.")

st.info("""
**Kategori Sentimen:**
* 🟢 **Positif:** Pro-lingkungan / Mendukung aksi iklim.
* 🔴 **Negatif:** Skeptis / Menyangkal pemanasan global.
* ⚪ **Netral:** Opini umum biasa.
* 🔵 **Berita:** Berbagi tautan fakta objektif.
""")

user_input = st.text_area("Ketik cuitan (Bahasa Inggris) di sini:", height=150)

if st.button("Analisis Sentimen"):
    if user_input.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong!")
    else:
        with st.spinner("AI sedang menganalisis..."):
            input_vektor = vectorizer.transform([user_input])
            hasil_prediksi = model.predict(input_vektor)[0]
            
            st.success("Selesai!")
            if "Positif" in hasil_prediksi:
                st.success(f"### Hasil: {hasil_prediksi}")
            elif "Negatif" in hasil_prediksi:
                st.error(f"### Hasil: {hasil_prediksi}")
            elif "Berita" in hasil_prediksi:
                st.info(f"### Hasil: {hasil_prediksi}")
            else:
                st.warning(f"### Hasil: {hasil_prediksi}")

st.caption("Proyek UAS Machine Learning | Naive Bayes")