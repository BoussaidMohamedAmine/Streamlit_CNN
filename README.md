# Application Streamlit — Intel Image Classification

Choisir un modèle (CNN de base, CNN amélioré, CNN + Data Augmentation), charger une image, voir la classe prédite et la confiance de chaque classe. Le bouton « Évaluer sur le jeu de test » calcule l'accuracy et la précision/rappel par classe du modèle choisi sur `data/seg_test`.

## Prérequis

Les modèles `models/*.keras` sont créés par `notebook.ipynb`. Exécutez le notebook en entier avant de lancer l'application.

## Lancement (depuis la racine du projet)

```bash
conda activate ai-env            # ou : pip install -r app/requirements.txt
streamlit run app/app.py
```
