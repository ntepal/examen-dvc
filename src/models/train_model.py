# src/models/train_model.py
# Récupération des paramètres sauvegardés pour entraîner le modèle final

import joblib
from sklearn.ensemble import RandomForestRegressor

# Import relatif au projet pour le batch à partir de examen-dvc
# NB: export PYTHONPATH=$PYTHONPATH:. à mettre dans le .sh qui l'appelle
import src.data.utilitaires as utilitaires

def train_final_model():
    #--- Vérification de la présence des répertoires sinon création
    utilitaires.check_else_create_dirs()

    #--- Chargement des données de train (X normalisé, y brut)
    X_train, y_train = utilitaires.load_processed_data(subset='train', scaled=True)

    #--- Chargement des hyperparamètres optimisés
    try:
        params = joblib.load('models/best_params.pkl')
        print(f"📖 Meilleurs paramètres chargés : {params}")
    except FileNotFoundError:
        print("⚠️ 'best_params.pkl' introuvable. Utilisation des paramètres par défaut.")
        params = {}

    #--- Création et entraînement du modèle final
    # **params pour injecter le dictionnaire directement dans les arguments
    model = RandomForestRegressor(**params, random_state=42)

    print("🧠 Entraînement du modèle final en cours...")
    model.fit(X_train, y_train)

    #--- Sauvegarde du modèle entraîné
    joblib.dump(model, 'models/trained_model.pkl')

    print("✅ Modèle final entraîné et sauvegardé dans models/trained_model.pkl")

if __name__ == "__main__":
    train_final_model()
