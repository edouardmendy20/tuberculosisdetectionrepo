import os
import joblib as jb
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------
# Configuration de la page (doit être la première commande Streamlit)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Portail IA - VGG16 & DNN",
    page_icon="🤖",
    layout="centered",
)

# ---------------------------------------------------------------
# Chargement des modèles et artefacts (mis en cache)
# ---------------------------------------------------------------
@st.cache_resource
def load_vgg16():
    path = "vgg16_model_finetuned.h5"
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path, compile=False)


@st.cache_resource
def load_dnn():
    files = ["DNN_model.h5", "encoders.joblib", "scaler.joblib"]
    if not all(os.path.exists(f) for f in files):
        return None, None, None
    model = tf.keras.models.load_model("DNN_model.h5", compile=False)
    encoders = jb.load("encoders.joblib")
    scaler = jb.load("scaler.joblib")
    return model, encoders, scaler


# ---------------------------------------------------------------
# Fonctions de prédiction
# ---------------------------------------------------------------
def predict_tuberculosis(model, image):
    classes = ["Normal", "Tuberculosis"]
    image = image.convert("RGB")
    image = image.resize((150, 150), Image.NEAREST)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    p_tb = float(model.predict(image_array, verbose=0)[0][0])
    return {classes[1]: p_tb, classes[0]: 1.0 - p_tb}


# ---------------------------------------------------------------
# Interface principale avec 2 onglets sur la même page
# ---------------------------------------------------------------
st.title("🤖 Plateforme de Prédiction IA")
st.write("Sélectionnez le modèle à utiliser ci-dessous :")

tab_vgg, tab_dnn = st.tabs([
    "🫁 Modèle 1 : Tuberculose (VGG16)",
    "🏦 Modèle 2 : Bank Marketing (DNN)",
])

# ===============================================================
# ONGLET 1 : VGG16 (Radiographie thoracique)
# ===============================================================
with tab_vgg:
    st.header("Détection de la tuberculose sur radiographie thoracique")
    st.write("Téléversez une radiographie thoracique pour estimer la probabilité de tuberculose.")
    st.warning("Outil pédagogique : il ne remplace pas un diagnostic médical.")

    vgg_model = load_vgg16()
    if vgg_model is None:
        st.error("Fichier `vgg16_model_finetuned.h5` introuvable dans le dossier.")
        st.stop()

    uploaded_file = st.file_uploader(
        "Radiographie thoracique",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col_image, col_result = st.columns(2)

        with col_image:
            st.image(image, caption="Image téléversée")

        with col_result:
            with st.spinner("Analyse en cours..."):
                probas = predict_tuberculosis(vgg_model, image)

            prediction = max(probas, key=probas.get)
            st.subheader("Résultat")
            if prediction == "Tuberculosis":
                st.error(f"**{prediction}** ({probas[prediction]:.1%})")
            else:
                st.success(f"**{prediction}** ({probas[prediction]:.1%})")

            for classe, proba in sorted(probas.items(), key=lambda x: x[1], reverse=True):
                st.write(f"{classe} : {proba:.1%}")
                st.progress(proba)
    else:
        st.info("Aucune image téléversée pour le moment.")

# ===============================================================
# ONGLET 2 : DNN (Bank Marketing Prediction)
# ===============================================================
with tab_dnn:
    st.header("Bank Marketing Prediction")
    st.write("Renseignez les informations du client pour prédire s'il va souscrire au dépôt à terme.")

    dnn_model, encoders, scaler = load_dnn()
    if dnn_model is None:
        st.error("Fichiers manquants (`DNN_model.h5`, `encoders.joblib` ou `scaler.joblib`).")
        st.stop()

    with st.form("bank_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
            job = st.selectbox("Job", list(encoders[0].classes_))
            marital = st.selectbox("Marital", list(encoders[1].classes_))
            education = st.selectbox("Education", list(encoders[2].classes_))
            default = st.selectbox("Default", list(encoders[3].classes_))
            housing = st.selectbox("Housing", list(encoders[4].classes_))
            loan = st.selectbox("Loan", list(encoders[5].classes_))
            contact = st.selectbox("Contact", list(encoders[6].classes_))

        with col2:
            month = st.selectbox("Month", list(encoders[7].classes_))
            day_of_week = st.selectbox("Day of week", list(encoders[8].classes_))
            duration = st.number_input("Duration (secondes)", min_value=0, value=180, step=1)
            campaign = st.number_input("Campaign", min_value=1, value=1, step=1)
            pdays = st.number_input("Pdays", min_value=0, value=999, step=1)
            previous = st.number_input("Previous", min_value=0, value=0, step=1)
            poutcome = st.selectbox("Poutcome", list(encoders[9].classes_))

        submitted = st.form_submit_button("Prédire la souscription", use_container_width=True)

    if submitted:
        job_enc = encoders[0].transform([job])[0]
        marital_enc = encoders[1].transform([marital])[0]
        education_enc = encoders[2].transform([education])[0]
        default_enc = encoders[3].transform([default])[0]
        housing_enc = encoders[4].transform([housing])[0]
        loan_enc = encoders[5].transform([loan])[0]
        contact_enc = encoders[6].transform([contact])[0]
        month_enc = encoders[7].transform([month])[0]
        day_enc = encoders[8].transform([day_of_week])[0]
        poutcome_enc = encoders[9].transform([poutcome])[0]

        x_new = np.array([
            age, job_enc, marital_enc, education_enc, default_enc,
            housing_enc, loan_enc, contact_enc, month_enc, day_enc,
            duration, campaign, pdays, previous, poutcome_enc
        ], dtype=np.float32).reshape(1, -1)

        x_new = scaler.transform(x_new)
        proba = float(dnn_model.predict(x_new, verbose=0)[0][0])
        y_pred = int(np.round(proba))

        st.subheader("Résultat")
        if y_pred == 1:
            st.success(f"**Souscrire** (Probabilité : {proba:.1%})")
        else:
            st.warning(f"**Pas souscrire** (Probabilité de souscription : {proba:.1%})")

st.divider()
st.caption("DIT & AfriAI Solutions · Déploiement multi-modèles (VGG16 & DNN)")