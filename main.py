import streamlit as st
import urllib.request
import json
import os
import ssl

# Fonction pour autoriser les certificats auto-signés
def allowSelfSignedHttps(allowed):
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True)  # Nécessaire si un certificat auto-signé est utilisé.

# Configuration de l'interface
st.title("Prédictions Titanic - Modèle Azure ML")
st.write("Entrez les caractéristiques du passager pour prédire s'il survivrait au Titanic.")

# Champs pour les données d'entrée
st.header("Entrez les caractéristiques du passager :")

pclass = st.selectbox("Classe (Pclass)", [1, 2, 3], help="Classe du passager (1 = Première classe, 3 = Troisième classe)")
age = st.slider("Âge", 0, 100, 29, help="Âge du passager en années")
sibsp = st.number_input("Nombre de frères/soeurs/conjoints à bord (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Nombre de parents/enfants à bord (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Tarif du billet (Fare)", min_value=0.0, step=0.1, value=50.0)
sex = st.selectbox("Sexe (0 = Homme, 1 = Femme)", [0, 1], help="0 pour Homme, 1 pour Femme")
embarked = st.selectbox("Port d'embarquement (0 = Cherbourg, 1 = Queenstown, 2 = Southampton)", [0, 1, 2])

# Bouton pour effectuer la prédiction
if st.button("Prédire la survie"):
    # Données d'entrée au bon format
    data = {
        "Inputs": {
            "input1": [
                {
                    "Survived" : 1,
                    "Pclass": pclass,
                    "Age": age,
                    "SibSp": sibsp,
                    "Parch": parch,
                    "Fare": fare,
                    "Sex": sex,
                    "Embarked": embarked
                }
            ]
        }
    }
    # Convertir les données en chaîne JSON
    body = str.encode(json.dumps(data))

    # URL de l'endpoint
    url = 'http://b2461160-7983-4b89-81f9-16b35fd79c5f.westeurope.azurecontainer.io/score'

    # Clé d'authentification (Primary Key ou Secondary Key)
    api_key = 'x8Jk9Zob7KYxgxA8fdXFKlN7EumoxXTM'
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    # Ajouter les en-têtes pour l'appel
    headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key)}

    # Préparer et exécuter la requête
    req = urllib.request.Request(url, body, headers)
    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        print("Réponse :", result.decode('utf-8'))
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

    try:
        response = urllib.request.urlopen(req)
        result = response.read().decode("utf-8")
        prediction = json.loads(result)
        print("preddddd", prediction)

        # Affichage des résultats
        st.success(f"Résultat : {'Survivant' if prediction['Results']['WebServiceOutput0'][0] == 1 else 'Non survivant'}")

    except urllib.error.HTTPError as error:
        st.error(f"Erreur HTTP : {error.code}")
        st.error(f"Message : {error.read().decode('utf-8')}")
