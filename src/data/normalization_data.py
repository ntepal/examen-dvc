# src/data/normalization_data.py
# Utilisation de StandardScaler pour harmoniser les échelles.

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Import relatif au projet pour le batch à partir de examen-dvc
# NB: export PYTHONPATH=$PYTHONPATH:. à mettre dans le .sh qui l'appelle
import src.data.utilitaires as utilitaires

def run_normalization():
    #--- Vérification de la présence des répertoires sinon création
    utilitaires.check_else_create_dirs()

    #--- Chargement des données splittées (via utilitaires.py)
    # y_tain et y_test sont inutiles pour la normalisation
    X_train, _ = utilitaires.load_processed_data(subset='train', scaled=False)
    X_test, _ = utilitaires.load_processed_data(subset='test', scaled=False)

    #--- Normalisation
    scaler = StandardScaler()

    # On ne l'applique qu'aux colonnes qui ne sont pas des dates
    cols_to_scale = [c for c in X_train.columns if c not in ['Year', 'Month', 'Day', 'Hour']]

    # Fit_transform sur le train / Transform sur le test
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    #--- Sauvegarde des fichiers
    stockage_dir = 'data/processed'
    X_train.to_csv(f'{stockage_dir}/X_train_scaled.csv', index=False)
    X_test.to_csv(f'{stockage_dir}/X_test_scaled.csv', index=False)

    #--- Sauvegarde du scaler
    joblib.dump(scaler, 'models/scaler.pkl')

    print(f"✅ Données normalisées sauvegardées dans {stockage_dir} "
          "et scaler.pkl sauvegardé dans models/")

if __name__ == "__main__":
    run_normalization()
