# Printed Classification MLPs

### Methodology ###
- Given a pre-trained MLP pruning, QAT and weight sharing is applied.
- The pruning sparsity and the quantization level for the networks' inputs, weights and biases are found through a hw-aware DSE using the NSGA-II algorithm (https://pymoo.org/algorithms/moo/nsga2.html).

### Structure ###

* Six classification MLPs from the open-source UCI repository (https://archive.ics.uci.edu/ml/datasets.php) are examined :
- Redwine
- Whitewine
- Pendigits
- Vertebral (3C)
- Seeds

* **trained models** : The pretrained .joblib models
* **genetic_<dataset_name>** : The jupyter-notebook and the python scripts, datasets for the methodology
* **Verilogs** :  The generated verilog designs for each dataset
