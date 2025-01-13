# Importation des bibliothèques nécessaires
from azureml.core import Workspace, Experiment, Dataset, Environment
from azureml.core.run import Run
from azureml.train.automl import AutoMLConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import os

# Étape 1 : Connexion au Workspace Azure ML
print("Connexion au Workspace Azure ML...")
ws = Workspace.from_config(path="./config.json")  # Assurez-vous que config.json est dans le répertoire courant
print(f"Workspace Name: {ws.name}")

# Étape 2 : Chargement et exploration des données
print("Chargement des données Titanic...")
data_path = "titanic.csv"  # Remplacez par le chemin de votre fichier Titanic CSV
df = pd.read_csv(data_path)

# Affichage des premières lignes des données
print(df.head())

# Prétraitement des données
print("Prétraitement des données...")
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna('S', inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Division des données
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 3 : Entraînement du modèle avec AutoML
print("Configuration d'AutoML...")
automl_settings = {
    "iteration_timeout_minutes": 10,
    "experiment_timeout_hours": 1,
    "primary_metric": "accuracy",
    "n_cross_validations": 5,
}

automl_config = AutoMLConfig(
    task="classification",
    training_data=pd.concat([X_train, y_train], axis=1),
    label_column_name="Survived",
    **automl_settings
)

experiment_name = "titanic-experiment"
experiment = Experiment(ws, experiment_name)

print("Démarrage de l'expérience AutoML...")
run = experiment.submit(automl_config, show_output=True)

# Étape 4 : Récupération et évaluation du meilleur modèle
print("Récupération du meilleur modèle...")
best_run, fitted_model = run.get_output()
print("Évaluation sur les données de test...")
y_pred = fitted_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy du modèle : {accuracy}")

# Étape 5 : Enregistrement du modèle dans Azure ML
print("Enregistrement du modèle dans Azure ML...")
model_path = "outputs/best_model.pkl"
os.makedirs("outputs", exist_ok=True)

import joblib
joblib.dump(fitted_model, model_path)

registered_model = best_run.register_model(
    model_name="titanic_model",
    model_path=model_path,
    description="Meilleur modèle pour le dataset Titanic",
)

print(f"Modèle enregistré avec l'ID : {registered_model.id}")

# Étape 6 : Déploiement en tant qu'endpoint
print("Déploiement du modèle en tant qu'endpoint...")
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig

# Charger le modèle enregistré
model = Model(ws, "titanic_model")

# Configurer l'inférence
inference_config = InferenceConfig(
    entry_script="score.py",  # Créez un fichier `score.py` pour définir la logique d'inférence
    environment=Environment(name="AzureML-sklearn"),
)

# Configuration du déploiement
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Déployer
service = Model.deploy(
    workspace=ws,
    name="titanic-service",
    models=[model],
    inference_config=inference_config,
    deployment_config=aci_config,
)

service.wait_for_deployment(show_output=True)
print(f"Service déployé à l'adresse : {service.scoring_uri}")

# Étape 7 : Test de l'API déployée
print("Test du service déployé...")
import requests

test_data = X_test.iloc[:1].to_json(orient="records")
response = requests.post(
    url=service.scoring_uri,
    headers={"Content-Type": "application/json"},
    data=test_data,
)
print(f"Réponse du service : {response.json()}")
