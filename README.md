# Diagnostic COVID-19 — Classification (Clinical & Laboratory Data)

## Contexte
Ce projet vise à prédire le résultat du test **SARS-Cov-2** (positif vs négatif) à partir de variables cliniques/biologiques issues du dataset **“Diagnostic of COVID-2019 and its clinical spectrum”**.

L’approche suit un pipeline complet :
**EDA → Preprocessing → Modélisation → Optimisation**.

---

## Objectif
Construire un modèle de **classification binaire** permettant d’estimer si un patient est **infecté (1)** ou **non infecté (0)**.

Dans un contexte médical, l’objectif est de limiter au maximum les **faux négatifs**, donc j'ai porté une attention particulière au **Recall** (sensibilité).

---

## Dataset
> Le dataset etant un peu lourd j'ai pas pu le mettre le Guithub mais vous pouvez le retrouver via ce lien
- Source : Kaggle — [Diagnostic of COVID-2019 and its clinical spectrum](https://www.kaggle.com/datasets/einsteindata4u/covid19)
- Licence : voir la page Kaggle du dataset
- Stockage local attendu : `data/`
- Taille initiale : **5644 lignes × 111 colonnes**
- Classes : **négatif majoritaire** (fort déséquilibre)
- Forte proportion de valeurs manquantes → filtrage et sélection de variables nécessaires.

> Remarque : après nettoyage (suppression des colonnes avec plus de 90% de NaN) le dataset est réduit à **39 colonnes**.  
> Après preprocessing (gestion NaN actuelle par suppression de lignes), l’échantillon utilisé pour l’entraînement/test est fortement réduit. 
---

## Méthodologie

### 1) EDA (Exploratory Data Analysis)
- Analyse de forme : dimensions, types, valeurs manquantes, distribution de la target
- Analyse de fond : exploration par sous-groupes (blood / viral), visualisations
- Tests statistiques (T-test) sur certaines variables sanguines

### 2) Preprocessing
- Encodage des variables catégorielles (`positive/negative`, `detected/not_detected`)
- Feature engineering : création d’une variable synthétique à partir des variables virales
- Gestion des valeurs manquantes (stratégie actuelle : suppression des lignes contenant des NaN)
- Séparation train/test

### 3) Modélisation & évaluation
Modèles testés (via pipelines) :
- RandomForestClassifier
- AdaBoostClassifier
- SVC (SVM)
- KNeighborsClassifier

Métriques :
- Precision
- Recall (prioritaire)
- F1-score
- Matrice de confusion
- Learning curves

### 4) Optimisation
- RandomizedSearchCV sur le pipeline SVM (scoring = recall, CV=4)
- Recherche d’hyperparamètres : `C`, `gamma`, degré PolynomialFeatures, `k` pour SelectKBest, etc.

---

## Résultats (jeu de test)
Les modèles atteignent une bonne performance sur la classe négative.
La difficulté principale reste la **détection des positifs** (Recall classe 1 encore limité), ce qui est cohérent avec:
- le déséquilibre de classes
- la forte perte de données après suppression des NaN

> J'ai retenu SVM comme modèle principal puis optimisé, avec des gains limités sur le test final.

---

## Installation & exécution

### Prérequis
- Python 3.9+ (recommandé)
- Jupyter Notebook / JupyterLab

### Librairies
- numpy, pandas, matplotlib, seaborn
- scikit-learn
- scipy

### Lancer le notebook
```bash
jupyter notebook
