## `ANKLES`: *A*mortized *N*ormalizing flows for *K*eplerian orbita*L* *E*stimation with *S*BI

Work in progress for a class!
---

### What is in this repository? 

`data/` : store training/test data
`models/` : store trained models
`plots/` : plots for my poster/paper

__notebooks__:

`generate_data_with_COSMIC.ipynb` : exactly what it sounds like. generating the train/test data using a binary population from the `COSMIC` code ([Breivik et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...898...71B/abstract)).

`train_baseline.ipynb` : train a "vanilla" MLP to estimate orbital parameters. It just does OK, and we show the need for probabilistic methods. Resulting MLP is stored in `models/`.

`train_SBI.ipynb` : Where I initially trained the SBI. Resulting model is stored in `models/`. 

`ankles.py` : (to be written) end-to-end inference from un-normalized inputs to un-normalized posteriors for ease of use.

`demo.ipynb` : (to be written) demonstration of using `ankles.py` to generate posteriors on your new data.