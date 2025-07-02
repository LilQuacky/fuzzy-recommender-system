### Prima prova: implementazione di base con Fuzzy C-Means

Come primo step, è stata implementata una versione base di un sistema di raccomandazione collaborativo basato su 
clustering fuzzy, seguendo l’approccio presentato in \[Koohi et al., 2016]. A partire dal dataset **MovieLens 100k**, 
è stata costruita la matrice utente-item e applicato l’algoritmo **Fuzzy C-Means (FCM)** per raggruppare gli utenti in 
cluster sovrapposti, permettendo quindi a ciascun utente di appartenere a più gruppi di preferenza con un grado di 
appartenenza fuzzy. Sulla base di questi cluster, è stato sviluppato un meccanismo di predizione dei rating mediando i 
valori dei centroidi, pesati per il grado di membership dell’utente.

Per valutare il modello, sono stati utilizzati gli indicatori di accuratezza **RMSE (3.6991)** e **MAE (3.5242)**, che 
hanno evidenziato una bassa qualità predittiva, con errori medi molto elevati rispetto al range dei rating (1–5). 
Inoltre, il valore di **entropia fuzzy (1.6094)** – pari al massimo teorico per 5 cluster – suggerisce che il clustering
ottenuto è altamente sfocato: gli utenti sono distribuiti in modo quasi uniforme su tutti i cluster, segno di una debole
struttura latente nei dati secondo FCM. Anche il plot risultante da una proiezione PCA ha mostrato due regioni 
principali, ma senza una chiara separazione netta, indicando la necessità di metodi fuzzy più espressivi o di strategie 
ibride per migliorare l'efficacia del sistema.

![Fuzzy C-Means Clustering](images/fcm_clusters_1.png)

### Seconda prova: estensione con modello ibrido Fuzzy C-Means + SVD

Per superare le limitazioni riscontrate nella prima implementazione, è stata introdotta una versione ibrida che 
combina il clustering fuzzy con una tecnica di riduzione della dimensionalità basata su **Truncated Singular Value
Decomposition (SVD)**. In questo modello, gli utenti sono raggruppati con FCM come in precedenza, ma all’interno di 
ciascun cluster la matrice di rating viene approssimata tramite SVD per catturare le caratteristiche latenti più 
rilevanti e migliorare la qualità della predizione.

I risultati mostrano che il modello ibrido peggiora le metriche di accuratezza rispetto al semplice FCM, con un 
**RMSE di 3.6971** e **MAE di 3.5126** sul test set. Questo suggerisce che l’integrazione diretta di SVD sui
sottoinsiemi fuzzy di utenti, così come implementata, non è efficace nel migliorare la predizione. La sovrapposizione 
fuzzy dei cluster probabilmente introduce rumore nella ricostruzione, evidenziando la necessità di un’integrazione più
raffinata tra clustering e decomposizione.

Dal confronto tra i due modelli, emerge che:

* Il clustering fuzzy da solo, pur mostrando performance modeste, garantisce una generalizzazione più stabile tra training e test.
* L’approccio ibrido richiede ulteriori affinamenti, come tuning dei parametri o l’adozione di metodi ibridi più sofisticati, per sfruttare appieno la complementarietà tra membership fuzzy e tecniche di estrazione delle caratteristiche latenti.

![Hybrid Fuzzy C-Means + SVD Clustering](images/fuzzy_membership_2.png)

### Terza prova: miglioramento pipeline

In questa fase, è stata introdotta una normalizzazione dei rating per utente, sottraendo la media personale per 
eliminare il bias di scala individuale. Successivamente, il clustering fuzzy C-Means è stato applicato sui dati 
normalizzati e standardizzati, migliorando la coerenza della struttura latente rilevata.

Il modello è stato valutato sia sul training set sia sul test set, utilizzando RMSE e MAE per misurare la qualità 
della predizione sui rating normalizzati. I risultati ottenuti sono stati **TRAIN RMSE: 0.2593, MAE: 0.0523** e 
**TEST RMSE: 0.2568, MAE: 0.0514**, evidenziando una bassa discrepanza tra i due insiemi e quindi una buona capacità 
di generalizzazione. Questo indica che la normalizzazione ha reso i dati più adatti al clustering fuzzy, migliorando 
la qualità predittiva rispetto alla versione base.

Il miglioramento è significativo rispetto ai risultati iniziali ottenuti senza normalizzazione, che mostravano errori 
molto più alti e membership più uniformi (quasi casuali). La distribuzione delle membership fuzzy ora mostra una 
struttura più definita, come evidenziato dalla visualizzazione PCA, sebbene permanga una certa sovrapposizione tra 
cluster, tipica delle tecniche fuzzy.

Questo approccio fornisce quindi una base solida per ulteriori sviluppi, come l’integrazione di metodi ibridi o 
l’ottimizzazione di parametri del clustering e della predizione.

### Quarta prova: sistemo cose

DONE:
* Valutazione su dati denormalizzati: normalizzi e predici valori normalizzati, ma i valori finali dovrebbero essere riportati alla scala originale per essere interpretabili
* Confronto con baseline: manca un confronto con altri metodi (ad esempio k-means hard clustering, o metodi di raccomandazione classici come kNN o matrix factorization).
* Test/Train Split: Stai dividendo gli utenti in train/test, ma nella realtà spesso si divide valutazioni per utente (ad esempio 80% delle valutazioni di ogni utente per train, 20% per test).
Separare completamente gli utenti può essere più difficile e meno realistico, perché nuovi utenti nel test non hanno dati in train (cold start utente).
* Predizione con soglia membership: la scelta della soglia (0.4) è arbitraria. Potresti sperimentare diverse soglie o metodi di aggregazione più sofisticati (es. ponderazioni continue senza soglia).

TODO:

1. [ ] Rimozione NaN: per la valutazione usi un semplice mask per valori non NaN, ma non consideri che alcune predizioni potrebbero essere molto approssimative se l’utente ha poche valutazioni.
2. [ ] Scalatura e normalizzazione: applichi StandardScaler dopo la normalizzazione per utente, potrebbe andare bene, ma attenzione a non mescolare scale che potrebbero influire sull’interpretazione delle distanze nel clustering fuzzy.
3. [ ] Feature selection: usi direttamente la matrice utente-item, ma potresti valutare di ridurre dimensionalità (es. PCA o SVD) prima del clustering per migliorare robustezza e interpretabilità.
