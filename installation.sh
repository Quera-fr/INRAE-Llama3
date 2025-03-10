#!/bin/bash

# Arrêter le script en cas d'erreur
set -e

echo "🚀 Début de l'installation..."

# 1. Création de l'environnement virtuel Python
echo "📦 Création de l'environnement virtuel '.llama_env'..."
python -m venv .llama_env

# Activation de l'environnement virtuel
# source .llama_env/bin/activate (sur mac ou linux)
source .llama_env/Scripts/activate

# 2. Téléchargement et installation des dépendances
echo "📥 Installation des dépendances nécessaires..."
pip install -r requirements.txt

# 3. Téléchargement de llama.cpp depuis GitHub
echo "🔄 Clonage de llama.cpp..."
git clone https://github.com/ggerganov/llama.cpp.git

echo "✅ Installation terminée !"
echo "🔹 Activez votre environnement avec : source llama_env/bin/activate"


# 4. Connexion à HuggingFace
echo "🔗 Connexion à HuggingFace..."
huggingface-cli login 

# Token : ...


# COnversion des modèles au format gguf
