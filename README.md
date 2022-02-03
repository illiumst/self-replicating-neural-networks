# Bureaucratic Cohort Swarms
### (The Meta-Task Experience)  # Deadline: 28.02.22
## Experimente
### Fixpoint Tests:
    
- [ ] Dropout Test 
  - (Macht das Partikel beim Goal mit oder ist es nur SRN)
  - Zero_ident diff = -00.04999637603759766 %
	   
- [ ] gnf(1) -> Aprox. Weight
  - Übersetung in ein Gewichtsskalar
    - Einbettung in ein Reguläres Netz
	
- [ ] Übersetung in ein Explainable AI Framework
  - Rückschlüsse auf Mikro Netze
	
- [ ] Visualiserung
  - Der Zugehörigkeit 
  - Der Vernetzung
	
- [ ] PCA()
  - Dataframe Epoch, Weight, dim_1, ..., dim_n
  - Visualisierung als Trajectory Cube
	
- [ ] Recherche zu Makro Mikro Netze Strukturen 
  - gits das schon?
  - Hypernetwork?
  - arxiv: 1905.02898

---

### Tasks für Steffen:

- [x] Training mit kleineren GNs
  - Accuracy leidet enorm (_0.56_)
    ![image info](./figures/training_lineplot.png)
  - Es entstehen mehr SRNN
  - Der Dropout Effekt wird stärker (diff_ohne_SRNN = _0.0_)
    ![image info](./figures/dropout_stacked_barplot.png)
- [ ] Weiter Trainieren -> 500 Epochs?
- [ ] Loss Gewichtung anpassen
- [ ] Training ohne Residual Skip Connection | - Running
- [ ] Test mit Baseline Dense Network 
  - [ ] mit vergleichbaren Neuron Count
  - [ ] mit gesamt Weight Count
- [ ] Task/Goal statt SRNN-Task

---

### Für Menschen mit zu viel Zeit:
- [ ] Sparse Network Training der Self Replication
  - Just for the lulz and speeeeeeed)
  - (Spaß bei Seite, wäre wichtig für schnellere Forschung)
    <https://pytorch.org/docs/stable/sparse.html>

---
