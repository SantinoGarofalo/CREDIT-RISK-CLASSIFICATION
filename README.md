# CREDIT-RISK-CLASSIFICATION

# Credit Risk Classification (German Credit Dataset)

Questo progetto implementa un workflow completo di Machine Learning supervisionato applicato al problema di credit risk classification, utilizzando il dataset German Credit disponibile su OpenML.

L’obiettivo è costruire, confrontare e interpretare diversi modelli di classificazione per prevedere il rischio di default di un cliente.

---

## Obiettivo

Dato un insieme di informazioni relative a un cliente, il modello deve predire se il cliente appartiene alla classe:

- Good (0) = buon pagatore
- Bad (1) = cattivo pagatore

La variabile target viene codificata come:

- Risk_binary = 1 se Risk == "bad"
- Risk_binary = 0 se Risk == "good"

In questo progetto la classe positiva è Bad (1).

---

## Dataset

Fonte: OpenML  
Nome dataset: credit-g  
Numero osservazioni: circa 1000  
Feature: variabili numeriche e categoriche  

Il dataset viene caricato tramite fetch_openml.

---

## Preprocessing

Il preprocessing è gestito tramite Pipeline e ColumnTransformer:

- Variabili numeriche: StandardScaler
- Variabili categoriche: OneHotEncoder(handle_unknown="ignore")

---

## Modelli valutati

Sono stati confrontati diversi modelli di classificazione:

- Logistic Regression (con class_weight="balanced")
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

---

## Strategia di validazione

Il dataset è stato suddiviso in:

- Train
- Validation
- Test

Per confrontare i modelli è stata inoltre utilizzata una Stratified K-Fold Cross Validation (5 fold).

Le metriche riportate nel confronto tra modelli rappresentano la media sui fold di cross-validation, con relativa deviazione standard.

---

## Metriche utilizzate

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Sono inoltre state prodotte:

- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

La valutazione finale del modello selezionato viene effettuata sul test set hold-out.

---

## Gestione dello sbilanciamento di classe

Nel dataset German Credit la classe Bad (1) è minoritaria rispetto alla classe Good (0).

In questo progetto si è scelto consapevolmente di non applicare tecniche di oversampling sul training set (ad esempio SMOTE, ROSE o Random Oversampling).

Motivazioni principali:

1. In un contesto reale di credit risk, la percentuale di default è spesso ancora più sbilanciata rispetto a quella presente nel dataset.
2. Tecniche di oversampling possono alterare la distribuzione della target e rendere meno affidabili le probabilità stimate dal modello.

Lo sbilanciamento viene gestito tramite:

- class_weight="balanced" nei modelli lineari
- metriche robuste allo sbilanciamento (ROC-AUC, F1-score)
- analisi Precision/Recall

---

## Hyperparameter tuning

Per il modello selezionato (XGBoost) è stato effettuato un tuning automatico tramite Optuna, ottimizzando gli iperparametri principali:

- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree
- min_child_weight
- gamma
- reg_lambda
- reg_alpha

L’ottimizzazione è guidata dalla metrica ROC-AUC.

---

## Explainability

Il progetto include interpretabilità del modello tramite:

- LIME (spiegazioni locali)
- SHAP (feature importance globale e locale)

---

## Esecuzione

Il notebook è pensato per essere eseguito in Google Colab.

Dipendenze principali:

- numpy
- pandas
- scikit-learn
- optuna
- xgboost
- lightgbm
- shap
- lime

---

## Autore

Santino Garofalo  
AI/ML Engineer
