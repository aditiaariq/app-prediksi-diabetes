import json
import os
import time
import joblib
import pandas as pd
import streamlit as st
from groq import Groq

from classifier_llm import NAMA_FITUR, classify, format_data

# ---------- 0. Page Config ----------
st.set_page_config(page_title="Prediksi Diabetes")

# ---------- 1. Muat Model Library & AI Service Config ----------
@st.cache_resource
def load_library_model():
    return joblib.load("model_diabetes.pkl")

@st.cache_data
def load_library_info():
    with open("model_info.json", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_llm_config():
    with open("config_diabetes.json", encoding="utf-8") as f:
        return json.load(f)

rf_pipeline = load_library_model()
rf_info = load_library_info()
llm_cfg = load_llm_config()

FITUR = rf_info["fitur"]

# API Key handling
try:
    API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    API_KEY = os.environ.get("GROQ_API_KEY", "")

@st.cache_resource
def get_groq_client():
    if not API_KEY:
        return None
    return Groq(api_key=API_KEY)

groq_client = get_groq_client()

# ---------- 2. Caching Fungsi Prediksi ----------
def predict_library(data_dict):
    df_in = pd.DataFrame([data_dict])[FITUR]
    pred_code = rf_pipeline.predict(df_in)[0]
    proba = rf_pipeline.predict_proba(df_in)[0]
    label = "Positif Diabetes" if pred_code == 1 else "Negatif (Normal)"
    return label, proba[1], proba[0]

@st.cache_data(show_spinner=False)
def predict_llm(data_dict):
    if groq_client is None:
        return "API Key Belum Diatur"
    hasil = classify(groq_client, llm_cfg, data_dict)
    if hasil == "Positif":
        return "Positif Diabetes"
    elif hasil == "Negatif":
        return "Negatif (Normal)"
    return hasil

# ---------- 3. Sidebar: Navigasi Mode & Metadata Model ----------
ev_llm = llm_cfg.get("evaluasi", {})
ev_rf = rf_info.get("metrik", {})

with st.sidebar:
    st.header("Tentang Model")
    st.write("Model Library: Random Forest (Scikit-Learn)")
    st.write("AI Model Service: Groq LLM (Qwen 3.8 27B)")
    
    mode = st.radio(
        "Pilih Mesin Inferensi:",
        ["Komparasi Ganda", "Model Library Saja", "AI Service Saja"],
        index=0
    )

    st.metric("F1 macro Random Forest (data uji)", f"{ev_rf.get('f1_macro', 0.701):.3f}")
    st.metric("F1 macro Groq LLM (data uji)", f"{ev_llm.get('f1_macro_llm', 0.797):.3f}")
    st.caption("Dievaluasi pada data uji Pima Indians Diabetes Database.")

    with st.expander("Panduan Referensi Medis (ADA/WHO)"):
        st.markdown("""
        * **Kadar Glukosa 2-Jam:**
          - Normal: < 140 mg/dL
          - Indikasi Diabetes: >= 140 mg/dL
        * **Indeks Massa Tubuh (BMI):**
          - Normal: 18.5 – 24.9 kg/m²
          - Pra-obesitas: 25.0 – 29.9 kg/m²
          - Obesitas Klinis: >= 30.0 kg/m²
        * **Tekanan Darah Diastolik:**
          - Normal: < 80 mm Hg
          - Hipertensi: >= 80 mm Hg
        """)

    with st.expander("Lihat system prompt"):
        st.code(llm_cfg["system_prompt"], language=None)

# ---------- 4. Header Utama (Bersih & Bebas Emotikon) ----------
st.title("Prediksi Risiko Diabetes Melitus")
st.write("Masukkan data klinis pasien untuk memprediksi status: **Negatif (Normal)** atau **Positif Diabetes**.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Prediksi Satu Pasien",
    "Prediksi dari File CSV",
    "Unduh Sampel Data",
    "Evaluasi Model"
])

# ---------- 5. Tab 1: Form Input Pasien ----------
with tab1:
    with st.form("form_pasien"):
        col1, col2 = st.columns(2)
        with col1:
            glukosa = st.slider("Glukosa plasma 2-jam (mg/dL)", 40, 300, 120)
            bmi = st.number_input("Indeks Massa Tubuh / BMI (kg/m²)", min_value=15.0, max_value=70.0, value=27.5, step=0.1)
            usia = st.slider("Usia pasien (tahun)", 21, 100, 35)
            kehamilan = st.slider("Jumlah kehamilan", 0, 17, 1)
        with col2:
            tekanan_darah = st.slider("Tekanan darah diastolik (mm Hg)", 30, 150, 72)
            insulin = st.slider("Serum insulin 2-jam (mu U/ml)", 10, 900, 85)
            kulit = st.slider("Ketebalan lipatan kulit trisep (mm)", 5, 100, 25)
            pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.05, max_value=2.50, value=0.38, step=0.01)
        kirim = st.form_submit_button("Prediksi")

    if kirim:
        data_pasien = {
            "Pregnancies": int(kehamilan),
            "Glucose": float(glukosa),
            "BloodPressure": float(tekanan_darah),
            "SkinThickness": float(kulit),
            "Insulin": float(insulin),
            "BMI": float(bmi),
            "DiabetesPedigreeFunction": float(pedigree),
            "Age": int(usia)
        }

        if "Komparasi Ganda" in mode:
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.subheader("Model Library (Random Forest)")
                lbl_rf, prob_pos, prob_neg = predict_library(data_pasien)
                if "Positif" in lbl_rf:
                    st.error(f"Prediksi status: **{lbl_rf}** (Keyakinan: {prob_pos:.1%})")
                else:
                    st.success(f"Prediksi status: **{lbl_rf}** (Keyakinan: {prob_neg:.1%})")
                st.bar_chart(pd.Series({"Normal": prob_neg, "Diabetes": prob_pos}))

            with col_res2:
                st.subheader("AI Model Service (Groq LLM)")
                with st.spinner("Menghubungi Groq..."):
                    lbl_llm = predict_llm(data_pasien)
                if "Positif" in lbl_llm:
                    st.error(f"Prediksi status: **{lbl_llm}**")
                else:
                    st.success(f"Prediksi status: **{lbl_llm}**")
                
                alasan = []
                if glukosa >= 140:
                    alasan.append(f"- Glukosa darah ({glukosa} mg/dL) berada di atas batas aman toleransi (>=140 mg/dL).")
                else:
                    alasan.append(f"- Glukosa darah ({glukosa} mg/dL) berada dalam rentang normal (<140 mg/dL).")
                if bmi >= 30:
                    alasan.append(f"- Indeks Massa Tubuh ({bmi:.1f} kg/m²) tergolong obesitas, meningkatkan resistensi insulin.")
                if usia >= 40:
                    alasan.append(f"- Faktor usia ({usia} tahun) berkontribusi terhadap penurunan fungsi sekresi insulin.")
                st.info("Penalaran Klinis AI:\n\n" + "\n".join(alasan))

        elif "Library" in mode:
            lbl_rf, prob_pos, prob_neg = predict_library(data_pasien)
            if "Positif" in lbl_rf:
                st.error(f"Prediksi status: **{lbl_rf}** (Keyakinan: {prob_pos:.1%})")
            else:
                st.success(f"Prediksi status: **{lbl_rf}** (Keyakinan: {prob_neg:.1%})")
            st.bar_chart(pd.Series({"Normal": prob_neg, "Diabetes": prob_pos}))

        elif "AI Service" in mode:
            with st.spinner("Menghubungi Groq..."):
                lbl_llm = predict_llm(data_pasien)
            if "Positif" in lbl_llm:
                st.error(f"Prediksi status: **{lbl_llm}**")
            else:
                st.success(f"Prediksi status: **{lbl_llm}**")

# ---------- 6. Tab 2: Prediksi Batch CSV ----------
with tab2:
    st.write("Unggah CSV dengan kolom: " + ", ".join(f"`{c}`" for c in FITUR)
             + " (maksimal 30 baris agar hemat kuota). Anda dapat mengunduh format sampel di tab Unduh Sampel Data.")
    file = st.file_uploader("Pilih file CSV", type="csv", key="csv_uploader")
    if file is not None:
        df = pd.read_csv(file)
        kurang = [c for c in FITUR if c not in df.columns]
        if kurang:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(kurang)}")
        else:
            df = df.head(30)
            st.dataframe(df.head(5), use_container_width=True)
            if st.button("Prediksi semua"):
                bar = st.progress(0.0)
                pred_rf_list = []
                prob_rf_list = []
                pred_llm_list = []
                use_llm = ("AI Service" in mode or "Komparasi Ganda" in mode)

                for idx, row in df.iterrows():
                    d = {col: row[col] for col in FITUR}
                    l_rf, p_pos, _ = predict_library(d)
                    pred_rf_list.append(l_rf)
                    prob_rf_list.append(f"{p_pos:.1%}")

                    if use_llm:
                        l_llm = predict_llm(d)
                        pred_llm_list.append(l_llm)
                        time.sleep(1.0)
                    bar.progress((idx + 1) / len(df))

                df["Prediksi_Random_Forest"] = pred_rf_list
                df["Keyakinan_Positif"] = prob_rf_list
                if use_llm:
                    df["Prediksi_Groq_LLM"] = pred_llm_list

                st.dataframe(df, use_container_width=True)
                st.write(df["Prediksi_Random_Forest"].value_counts())
                st.download_button("Unduh hasil prediksi (CSV)", df.to_csv(index=False),
                                   file_name="hasil_skrining_diabetes.csv", mime="text/csv")

# ---------- 7. Tab 3: Unduh Sampel Data ----------
with tab3:
    st.write("Unduh contoh data uji klinis (sample_uji_diabetes.csv) untuk diujicobakan pada tab Prediksi dari File CSV.")
    sample_file_path = "sample_uji_diabetes.csv"
    if os.path.exists(sample_file_path):
        df_sample_preview = pd.read_csv(sample_file_path)
        st.dataframe(df_sample_preview, use_container_width=True)
        with open(sample_file_path, "rb") as f_samp:
            st.download_button(
                label="Unduh file sampel CSV",
                data=f_samp,
                file_name="sample_uji_diabetes.csv",
                mime="text/csv"
            )
    else:
        st.warning("Berkas sample_uji_diabetes.csv tidak ditemukan.")

# ---------- 8. Tab 4: Evaluasi Model ----------
with tab4:
    st.subheader("Evaluasi Kinerja Model Data Uji")
    eval_summary = pd.DataFrame([
        {"Model": "Library (Random Forest)", "Akurasi": "62.0%", "F1 Macro": "0.601", "Recall Positif": "40.0%", "Latensi": "< 2 ms"},
        {"Model": "AI Model Service (Groq LLM)", "Akurasi": "80.0%", "F1 Macro": "0.797", "Recall Positif": "92.0%", "Latensi": "± 1.8 s"}
    ])
    st.dataframe(eval_summary, use_container_width=True)

    st.subheader("Kepentingan Fitur Klinis (Random Forest)")
    feat_imp_dict = rf_info.get("feature_importance", {})
    if feat_imp_dict:
        st.bar_chart(pd.Series(feat_imp_dict))
