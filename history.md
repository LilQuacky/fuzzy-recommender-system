## 🎯 Obiettivo e originalità

* Gli autori propongono per la prima volta l’impiego del clustering fuzzy C‑means (FCM) nel contesto del collaborative filtering *user‑based*.
* Scopo: migliorare le prestazioni del CF rispetto a clustering classici come K‑means e Self‑Organizing Map (SOM), attenuando problemi di *sparsità* e *scalabilità*. ([researchgate.net][1])

---

## 🛠 Metodo proposto

1. **Pre‑processing dataset**

   * Utilizzo del dataset MovieLens (100 000 valutazioni, 943 utenti, 1682 film, densità \~6.3%) ([researchgate.net][1])
   * Validazione incrociata a 5 fold: 80% training, 20% test. ([researchgate.net][1])

2. **Clustering utenti**

   * Confronto tra tre metodi:

     * *K‑means*
     * *SOM*
     * *Fuzzy C‑Means* con approssimazione membership fuzzy
   * Numero di cluster variabile (3,5,7…15), per determinare configurazione ottimale. ([researchgate.net][1])

3. **Defuzzificazione**

   * Due strategie per assegnare utenti a cluster:

     * **Maximum**: massimo grado di appartenenza
     * **Center of Gravity (COG)**: media ponderata delle appartenenze
   * In seguito si selezionano utenti “vicini” usando soglia su Pearson. ([researchgate.net][1])

4. **Calcolo della previsione**

   * Due metodi: media semplice e media pesata tramite coefficiente di Pearson.&#x20;

---

## 📊 Risultati (MovieLens)

* **Metriche usate**: accuratezza, precision, recall. ([koreascience.or.kr][2])
* Configurazione migliore: **FCM + COG + Pearson**, con 3 cluster.

  * **Accuratezza** ≈ 81.41%, **Precision** ≈ 64.81%, **Recall** ≈ 19.01%. ([researchgate.net][1])
* In confronto: K‑means e SOM mostrano performance inferiori (accuracy \~80% o meno, precision \~58–62%). ([researchgate.net][1])
* Aumentare il numero di cluster oltre 3 porta a un calo nelle performance.&#x20;

---

## ✅ Punti di forza

* **Risultati empirici solidi**: FCM supera metodi convenzionali su dataset MovieLens.
* **Analisi sistematica**: confronto tra algoritmi (K‑means, SOM, FCM), defuzzificazioni, modalità di previsione.
* **Riduce problema di sparsità**: clustering fuzzy aiuta ad assegnare utenti a gruppi anche con dati limitati.
