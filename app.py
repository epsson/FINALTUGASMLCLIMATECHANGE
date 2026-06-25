import streamlit as st
import joblib

# Load model dan vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("🎓 Sistem Klasifikasi Keluhan Mahasiswa")
st.write("Masukkan laporan atau keluhan mahasiswa untuk dikategorikan secara otomatis.")

user_input = st.text_area("Tulis keluhan di sini:")

if st.button("Klasifikasikan"):
    if user_input:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        st.success(f"Kategori Keluhan: {prediction}")
    else:
        st.warning("Tolong masukkan teks keluhan.")