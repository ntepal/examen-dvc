# src/models/GridSearch_RF_HyperParams.py
# Choix de RandomForestRegressor, très robuste pour ce type de données

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Import relatif au projet pour le batch à partir de examen-dvc
# NB: export PYTHONPATH=$PYTHONPATH:. à mettre dans le .sh qui l'appelle
import src.data.utilitaires as utilitaires

def tune_hyperparameters():
    #--- Vérification de la présence des répertoires sinon création
    utilitaires.check_else_create_dirs()

    #--- Chargement des données de train (X normalisé, y brut)
    X_train, y_train = utilitaires.load_processed_data(subset='train', scaled=True)

    #--- Configuration du modèle
    model = RandomForestRegressor(random_state=42)

    # Grille de paramètres à tester
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 15, 30],
        'min_samples_leaf': [1, 2]
    }

    #--- Lancement de la recherche (GridSearch)
    # n_jobs=-1 utilise tous les cœurs de l'instance Ubuntu
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='r2',
        verbose=1 # Optionnel : pour voir l'avancement sur la console Ubuntu
    )

    print("⏳ GridSearch en cours sur l'instance Ubuntu...")
    grid.fit(X_train, y_train)

    #--- Sauvegarde des meilleurs paramètres
    # On stocke le dictionnaire pour que train_model.py puisse le lire
    joblib.dump(grid.best_params_, 'models/best_params.pkl')

    print(f"✅ GridSearch terminé. Meilleurs paramètres : {grid.best_params_}")

if __name__ == "__main__":
    tune_hyperparameters()
