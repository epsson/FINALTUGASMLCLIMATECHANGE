import streamlit as st
import joblib

# Load model dan vectorizer
model = joblib.load('model_naive_bayes.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Kamus Penjelasan Kategori
deskripsi_kategori = {
    "Academic Support and Resources": "Keluhan ini berkaitan dengan akses materi kuliah, fasilitas riset, atau bimbingan dari dosen/akademik.",
    "Food and Cantines": "Keluhan ini mengenai kualitas makanan, kebersihan kantin, harga, atau perilaku staf kantin.",
    "Facilities and Infrastructure": "Keluhan terkait sarana prasarana kampus seperti Wi-Fi, gedung, ruang kelas, atau fasilitas lab.",
    "Administrative Services": "Keluhan tentang birokrasi pendaftaran, pembayaran UKT, atau layanan kartu mahasiswa.",
    "Student Life and Safety": "Keluhan mengenai aspek keamanan lingkungan kampus atau kenyamanan fasilitas pendukung lainnya."
}

st.title("🎓 Sistem Klasifikasi Keluhan Mahasiswa")
st.write("Masukkan keluhan mahasiswa untuk mengetahui kategori layanannya.")

user_input = st.text_area("Tulis keluhan di sini:")

if st.button("Klasifikasikan"):
    if user_input:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        
        # Tampilkan Hasil
        st.success(f"### Kategori Keluhan: {prediction}")
        
        # Tampilkan Deskripsi dari Kamus
        penjelasan = deskripsi_kategori.get(prediction, "Kategori tidak ditemukan dalam deskripsi.")
        st.info(f"**Penjelasan:** {penjelasan}")
    else:
        st.warning("Tolong masukkan teks keluhan.")