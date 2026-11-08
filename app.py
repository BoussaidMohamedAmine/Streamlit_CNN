"""
Application Streamlit — Intel Image Classification
Lancer depuis la racine du projet :   streamlit run app/app.py
Les modèles app/models/*.keras sont sauvegardés par notebook.ipynb.
"""
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ICI = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ICI, "models")
TEST_DIR = os.path.join(ICI, "..", "data", "seg_test", "seg_test")

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]   # ordre alphabétique = ordre Keras
IMG_SIZE = (150, 150)
MODELES = {
    "cnn_initial.keras": "CNN de base",
    "cnn_ameliore.keras": "CNN amélioré (Dropout + EarlyStopping)",
    "cnn_augmente.keras": "CNN Dense128 + Data Augmentation",
}
EMOJI = {"buildings": "🏢", "forest": "🌲", "glacier": "🧊", "mountain": "⛰️", "sea": "🌊", "street": "🛣️"}

st.set_page_config(page_title="Intel Image Classification", page_icon="🏞️", layout="wide")


@st.cache_resource(show_spinner="Chargement du modèle…")
def charger_modele(fichier):
    import tensorflow as tf  # import tardif : l'interface s'affiche plus vite
    return tf.keras.models.load_model(os.path.join(MODEL_DIR, fichier))


@st.cache_data(show_spinner="Évaluation sur le jeu de test (3 000 images)…")
def evaluer(fichier):
    """Accuracy + précision/rappel par classe sur data/seg_test, avec le même pipeline que le notebook."""
    import tensorflow as tf
    from sklearn.metrics import classification_report
    ds = tf.keras.utils.image_dataset_from_directory(TEST_DIR, image_size=IMG_SIZE, batch_size=32, shuffle=False)
    y_true = np.concatenate([y.numpy() for _, y in ds])
    y_pred = np.argmax(charger_modele(fichier).predict(ds, verbose=0), axis=1)
    return classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)


def pretraiter(img):
    # Même prétraitement que image_dataset_from_directory : RGB, redimensionnement bilinéaire, valeurs 0-255.
    # La normalisation /255 est faite par la couche Rescaling du modèle.
    img = img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    return np.expand_dims(np.asarray(img, dtype="float32"), 0)


st.title("🏞️ Intel Image Classification")
st.caption("Classification de scènes naturelles en 6 classes avec les CNN entraînés dans notebook.ipynb")

dispo = [f for f in MODELES if os.path.exists(os.path.join(MODEL_DIR, f))]
if not dispo:
    st.error("Aucun modèle trouvé dans `app/models/`. Exécutez d'abord `notebook.ipynb` en entier : "
             "il sauvegarde les modèles `.keras`.")
    st.stop()

# ---------------------------------------------------------------- barre latérale
with st.sidebar:
    st.header("⚙️ Modèle")
    choix = st.radio("Choisir le modèle", dispo, format_func=MODELES.get)

    st.subheader("📊 Précision du modèle")
    if os.path.isdir(TEST_DIR):
        if st.button("Évaluer sur le jeu de test", width="stretch"):
            st.session_state[f"eval_{choix}"] = evaluer(choix)
        rep = st.session_state.get(f"eval_{choix}")
        if rep:
            st.metric("Accuracy test", f"{rep['accuracy']:.2%}")
            st.dataframe(
                pd.DataFrame({c: {"Précision": rep[c]["precision"], "Rappel": rep[c]["recall"],
                                  "F1": rep[c]["f1-score"]} for c in CLASSES}).T.style.format("{:.3f}"),
                width="stretch",
            )
    else:
        st.caption("Dossier data/seg_test introuvable : évaluation indisponible.")
    st.divider()
    comparer = st.checkbox("Comparer avec tous les modèles", value=False)

# ---------------------------------------------------------------- image
fichier = st.file_uploader("📤 Charger une image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if fichier is None:
    st.info("Chargez une image de : " + ", ".join(f"{EMOJI[c]} {c}" for c in CLASSES))
    st.stop()

image = Image.open(fichier)
x = pretraiter(image)

col_img, col_res = st.columns([1, 1.3])
with col_img:
    st.image(image, caption=f"Image chargée ({image.size[0]}×{image.size[1]})", width="stretch")

probs = charger_modele(choix).predict(x, verbose=0)[0]
k = int(np.argmax(probs))

with col_res:
    st.subheader(f"Prédiction : {EMOJI[CLASSES[k]]} **{CLASSES[k]}**")
    st.metric("Confiance (score Softmax)", f"{probs[k]:.2%}")
    st.progress(float(probs[k]))
    if probs[k] < 0.5:
        st.warning("Confiance faible : l'image est ambiguë ou n'appartient peut-être à aucune des 6 classes.")

    df = pd.DataFrame({"Classe": CLASSES, "Probabilité": probs}).sort_values("Probabilité", ascending=False)
    st.bar_chart(df.set_index("Classe"), horizontal=True)
    st.dataframe(df.style.format({"Probabilité": "{:.2%}"}), hide_index=True, width="stretch")

# ---------------------------------------------------------------- comparaison
if comparer:
    st.divider()
    st.subheader("Comparaison des modèles sur cette image")
    lignes, colonnes = [], {}
    for f in dispo:
        p = charger_modele(f).predict(x, verbose=0)[0]
        j = int(np.argmax(p))
        lignes.append({"Modèle": MODELES[f], "Classe prédite": CLASSES[j], "Confiance": p[j]})
        colonnes[MODELES[f]] = p
    st.dataframe(pd.DataFrame(lignes).style.format({"Confiance": "{:.2%}"}), hide_index=True, width="stretch")
    st.bar_chart(pd.DataFrame(colonnes, index=CLASSES), stack=False)
