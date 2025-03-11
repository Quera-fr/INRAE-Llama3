# 🦙 Fine-Tuning du Modèle LLAMA 3 & Intégration Open-WebUI/RAG  

🚀 **Ce projet permet d'effectuer un fine-tuning du modèle LLAMA 3 et d'intégrer ses fonctionnalités avec Open-WebUI et un système RAG (Retrieval-Augmented Generation).**  

## 📂 Structure du Projet  

📌 **Fichiers et Dossiers**  

| Fichier/Dossier                 | Description |
|---------------------------------|------------|
| `Fine-tuning du modèle Llama3.ipynb` | Notebook principal pour le fine-tuning du modèle **LLAMA 3**. |
| `Fonctions Open-WebUI et RAGrag.ipynb` | Notebook contenant les fonctionnalités pour **Open-WebUI** et **RAG**. |
| `Modelfile` | Fichier de configuration pour l'intégration du modèle avec **Ollama**. |
| `Projet 1.ipynb` | Notebook pour une première application utilisant le modèle fine-tuné. |
| `Projet 2.ipynb` | Notebook pour une seconde application utilisant le modèle fine-tuné. |
| `data.jsonl` | **Dataset d'entraînement** utilisé pour le fine-tuning du modèle. |
| `installation.sh` | **Script d’installation** des dépendances et de configuration de l’environnement. |
| `requirements.txt` | Liste des dépendances nécessaires pour exécuter les notebooks. |

---

## 🏗️ **Installation et Configuration**  

### 1️⃣ **Cloner le projet**
```bash
git clone https://github.com/ton-repo/llama3-finetuning.git
cd llama3-finetuning
```

### 2️⃣ **Installation des Dépendances**
Avant de commencer, assurez-vous d’avoir **Python 3.12** et **Git** installés.  
Ensuite, exécutez le script d’installation :
```bash
bash installation.sh
```
Cela installera toutes les bibliothèques et téléchargera `llama.cpp`.

### 3️⃣ **Connexion à Hugging Face & Téléchargement du Modèle**
- **Créer un compte** [Hugging Face](https://huggingface.co/)
- **Obtenir un token d’accès** : [Créer un token ici](https://huggingface.co/settings/tokens)
- **Demander l'accès** au modèle **LLAMA 3.2-1B-Instruct** : [Accès au modèle](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- **Se connecter** dans le notebook :
```python
from huggingface_hub import login

HUGGINGFACE_TOKEN = input("🔑 Entrez votre token Hugging Face : ")
login(token=HUGGINGFACE_TOKEN)
```

---

## 🎯 **1. Fine-Tuning du Modèle LLAMA 3**
Le fine-tuning se fait dans **`Fine-tuning du modèle Llama3.ipynb`** :  
📌 **Étapes couvertes** :
- **Préparation du dataset** (`data.jsonl`) avec des questions sur **l'INRAE**.
- **Utilisation de LoRA** pour entraîner uniquement certaines couches du modèle.
- **Optimisation mémoire** pour un fine-tuning efficace même sans GPU.
- **Enregistrement du modèle fine-tuné** et fusion des poids.

---

## 🌍 **2. Intégration avec Open-WebUI et RAG**
Le fichier **`Fonctions Open-WebUI et RAGrag.ipynb`** permet d’utiliser le modèle fine-tuné avec :
- **Open-WebUI** : Interface pour tester et interagir avec le modèle.
- **RAG (Retrieval-Augmented Generation)** : Améliore les réponses en intégrant une base de connaissances.

---

## 🔄 **3. Conversion du Modèle pour Ollama**
Après le fine-tuning, nous convertissons le modèle pour une exécution optimisée avec **Ollama**.  
📌 **Fichier `Modelfile` :**
```bash
FROM ./llama3-finetuned-merged/Llama-3.2-1B-Instruct-F16.gguf
PARAMETER stop "<|endoftext|>"
```
📌 **Ajout du modèle dans Ollama :**
```bash
ollama create llama-inrae -f Modelfile
```
📌 **Test du modèle :**
```bash
ollama run llama-inrae "Quelles sont les champs d'activité de l'INRAE ?"
```

---

## 🛠 **4. Déploiement et Applications**
Les fichiers **`Projet 1.ipynb`** et **`Projet 2.ipynb`** sont des exemples d’utilisation du modèle fine-tuné dans différents cas d’application.

---

## ✅ **Résumé des Technologies Utilisées**
| 📌 Technologie | 💡 Utilisation |
|--------------|-------------|
| **LLAMA 3** | Modèle de langage pré-entraîné de **Meta AI**. |
| **LoRA** | Optimisation du fine-tuning pour économiser la mémoire. |
| **Hugging Face Transformers** | Gestion et entraînement du modèle. |
| **Ollama** | Exécution optimisée du modèle fine-tuné. |
| **Open-WebUI** | Interface utilisateur pour interagir avec le modèle. |
| **RAG** | Intégration d’une base de connaissances. |

---

## 🚀 **Contribuer au Projet**
Vous pouvez contribuer en :
- 🔹 **Améliorant les datasets**
- 🔹 **Optimisant le fine-tuning**
- 🔹 **Intégrant le modèle avec d'autres interfaces**

📩 **Contact** : *Quera - Kevin DURANTY  kevin.duranty@quera.fr*

---
