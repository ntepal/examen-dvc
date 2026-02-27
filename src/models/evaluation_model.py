# src/models/evaluation_model.py
# Calcul des métriques MSE, R², MAE et génération du fichier de prédictions

import joblib
import json
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Import relatif au projet pour le batch à partir de examen-dvc
# NB: export PYTHONPATH=$PYTHONPATH:. à mettre dans le .sh qui l'appelle
import src.data.utilitaires as utilitaires

def run_evaluation():
    #--- Vérification de la présence des répertoires sinon création
    utilitaires.check_else_create_dirs()

    #--- Chargement des données de test (X normalisé, y brut)
    X_test, y_test = utilitaires.load_processed_data(subset='test', scaled=True)

    #--- Chargement du modèle entraîné
    model = joblib.load('models/trained_model.pkl')

    #--- Prédictions
    print("🔮 Calcul des prédictions sur le jeu de test...")
    predictions = model.predict(X_test)

    #--- Calcul des métriques de performance
    metrics = {
        "mse": mean_squared_error(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions)
    }

    #--- Sauvegarde des métriques en JSON (pour DVC)
    with open('metrics/scores.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    #--- Sauvegarde des prédictions dans un CSV
    # On crée un dossier data/ s'il n'existe pas pour éviter les erreurs
    pd.DataFrame(predictions, columns=['predictions']).to_csv(
        'data/predictions.csv', index=False
    )

    print(f"✅ Évaluation terminée. R² Score: {metrics['r2']:.4f}")
    print("📊 Les scores sont dans metrics/scores.json et les prédictions dans data/predictions.csv")

if __name__ == "__main__":
    run_evaluation()
