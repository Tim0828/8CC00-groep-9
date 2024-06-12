# 8CC00 groep 9
 Notities bij Groepsopdracht 8CC00 drug discovery

## Wat te doen?
1. Github repository
2. Dataset preparation: 
    * ~~RDKit, hoe werkt het~~ _Tim_
    * ~~Dataset opbouwen van molecular discriptors~~ _Tim_
3. EDA: welke discriptors zijn relevant voor het model?
    * correlatie discriptor-inhibitie
    * _outlier analysis_
    * removing possible missing values
    * test assumptions
    * suggesties? (vul dan deze readme aan)
4. Identification of structure-activity relationships
    * decision trees
    * "logistic" regression 
    * clustering. (e.g. by kMeans)
5. Best approach: speed, accuracy
    * Deep learning model (for example with [keras](https://keras.io/getting_started/intro_to_keras_for_engineers/))
    * regression
    * decision tree
    * clustering 
6. prediction
7. report
8. presentation

+++ Notes Meeting 28-5 +++

Data cleaning en loading is gelukt.
PCA moet gebeuren voor verschillende onderdelen.
EDA: Nog niet heel veel aan gebeurt.
Clustering: 
•	Eerst preprocessing (minmax)
•	5 clusters op attributes.
•	Moet nog gevisualiseerd te worden.
.Regression:
•	Nog niet echt bekend welke attributen relevant zijn.
•	Er is nog geen code, maar wel een stappenplan van wat er moet gebeuren.
Deep learning:
•	Model getraind op de beschikbare data. (80% train, 20% test)
•	Accuracy is 0.93.
•	Je wilt dat je loss naar beneden gaat en op een gegeven moment ga je overfitten: dus hiervoor hebben we bijvoorbeeld early stopping toegevoegd.
•	Het is vooral uitproberen hiermee.

Taken:
Iedereen PCA voor volgende meeting, want het is redelijk nieuw concept voor iedereen. Dan kunnen we het vergelijken welke het beste is. Daarna kan iedereen gewoon aan zn eigen stukje werken.

++ Notes Meeting 04-06 ++

Iedereen gaat verder
Geen k-means, maar k nearest neighbour en RF
Beste approach? > liever false positive dan false negative, balanced accuracy?
Voor deap learning: slide over binary coefficient
Volgende week kiezen welke approach het beste is.

++ Notes Meeting 11-06 ++
Hiske & Julia\\
Random Forest op geselecteerde variables: 0.95-0.98\\
--> balanced accuracy berekenen voor de vergelijking\\
Levi & Duy\\
Ook random forest en k-NN. \\
--> balaced accuracy nog berekenen\\
Rosanne & Isabeau\\
Regressie, met geselecteerde kolommen, ook met gewichten geprobeerd (daar is alle kolommen beter): ~50% (pp uitgewerkt, nog samenvoegen?)\\
Isabeau nog PCA: andere PCA resultaten, even testen op modellen\\
Jelle & Tim\\
Fresh model: 0.97-0.99\\
Hyperparametertuning\\
## Wat gaan we doen??
Jelle: M&M \\
Iedereen: schrijf over eigen methode\\
Tabel balanced accuracies\\


