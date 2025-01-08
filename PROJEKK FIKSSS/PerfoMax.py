import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu

# Memuat model yang telah disimpan
tree = joblib.load('dt_tree_model.pkl')
forest = joblib.load('rf_forest_model.pkl')

# Fitur untuk input data
features = ['satisfaction_level', 'last_evaluation', 'average_montly_hours', 'time_spend_company', 'left']

# Prediksi fungsi
def predict(tree, data_point):
    for feature, branches in tree.items():
        value = data_point[feature]
        if value in branches:
            branch = branches[value]
            if isinstance(branch, dict):
                return predict(branch, data_point)
            else:
                return branch

def predict_forest(forest, data_point):
    predictions = [predict(tree, data_point) for tree in forest]
    return max(set(predictions), key=predictions.count)

def predict_voting(dt_tree, rf_forest, data_point):
    dt_prediction = predict(dt_tree, data_point)
    rf_prediction = predict_forest(rf_forest, data_point)
    combined_predictions = [dt_prediction, rf_prediction]
    return max(set(combined_predictions), key=combined_predictions.count)

# Halaman utama aplikasi dengan navbar
st.title("Aplikasi Prediksi Kinerja Karyawan")

# Navbar menu
def nav_menu():
    return option_menu(
        menu_title=None,  # required
        options=["Home Page", "Guide", "Predict"],  # required
        icons=["house", "info-circle", "bar-chart"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
    )

menu = nav_menu()

# Halaman Home
if menu == "Home Page":
    st.subheader("Selamat datang di Aplikasi PerfoMax!")
    st.write("Aplikasi ini digunakan untuk memprediksi kinerja karyawan berdasarkan beberapa fitur seperti tingkat kepuasan, evaluasi terakhir, jam kerja bulanan rata-rata, lama bekerja di perusahaan, dan apakah karyawan tersebut bertahan atau keluar.")
    st.write("Pilih menu di navbar untuk melihat panduan atau melakukan prediksi.")

# Halaman Guide
elif menu == "Guide":
    st.subheader("Panduan Penggunaan Aplikasi")
    st.write("1. Pilih menu 'Predict' untuk memulai prediksi.")
    st.write("2. Masukkan nilai untuk setiap fitur yang diminta.")
    st.write("3. Klik tombol 'Prediksi' untuk melihat hasil prediksi.")
    st.write("4. Hasil prediksi akan menunjukkan apakah karyawan tersebut memiliki kinerja yang baik atau buruk berdasarkan model yang digunakan.")

# Halaman Predict
elif menu == "Predict":
    st.subheader("Prediksi Kinerja Karyawan")

    # Input data dari pengguna dengan text input
    satisfaction_level = st.text_input("Satisfaction Level (0.0 to 1.0)", "0.5")
    last_evaluation = st.text_input("Last Evaluation (0.0 to 1.0)", "0.5")
    average_montly_hours = st.text_input("Average Monthly Hours (100 to 300)", "200")
    time_spend_company = st.text_input("Time Spend in Company (Years, 1 to 10)", "5")
    left = st.selectbox("Left (Yes/No)", ["ya", "tidak"])

    # Memastikan input valid dan mengonversi ke tipe data yang tepat
    try:
        satisfaction_level = float(satisfaction_level)
        last_evaluation = float(last_evaluation)
        average_montly_hours = int(average_montly_hours)
        time_spend_company = int(time_spend_company)
        left = 1 if left == 'ya' else 0
    except ValueError:
        st.error("Harap masukkan nilai yang valid!")

    # Membuat input data menjadi DataFrame
    data_input = {
        'satisfaction_level': satisfaction_level,
        'last_evaluation': last_evaluation,
        'average_montly_hours': average_montly_hours,
        'time_spend_company': time_spend_company,
        'left': left
    }

    data_input_df = pd.DataFrame([data_input], columns=features)

    # Tombol untuk melakukan prediksi
    if st.button("Prediksi"):
        # Prediksi menggunakan Voting Ensemble
        prediction = predict_voting(tree, forest, data_input_df.iloc[0])

        # Menampilkan hasil prediksi
        st.subheader("Hasil Prediksi:")
        st.write(f"Prediksi Kinerja: {prediction}")
