## ``ANKLES``: *A*mortized *N*ormalizing flows for *K*eplerian orbita*L* *E*stimation with *S*BI

`ANKLES` is a simulation-based inference (SBI) model to estimate the Keplerian orbital parameters $v_0, K, \omega, \phi_0, e, P$ from the radial velocity curves of single-lined spectroscopic binaries ("SB*1*s"... not to be confused with SBI...)

The goal is probabilistic estimates of the orbital parameters, which typically are achieved with time-consuming stochastic sampling techniques (e.g., [Price-Whelan et al. 2017](https://ui.adsabs.harvard.edu/abs/2017ApJ...837...20P/abstract)). One could implement an amortized variational inference network, but this requires expensive likelihood calculations to optimize during training using the Evidence Lower Bound (ELBo). Since RV curves are easy to forward model, we can generate a realistic simulated training set and, with SBI, circumvent the need for likelihood calculations.

Basically, no ELBOs here, only `ANKLES` ;)

![image](plots/machine_learns.png)
<!-- <img src="plots/machine_learns.jpg" alt="a machine learns" width="200"/> -->


### To-do list

I may have learned a lot so far but the machine has not learned nearly enough...

- [x] Deliver a mediocre poster presentation
- [ ] Train a model with RV curves that only have 3 visits (more plausible observing strategy)
- [ ] Train with _many_ more points (~1e6 as opposed to current 5e4)
- [ ] Train a model that "focuses" on shorter-period training data (from based on current model performance, less than P~1e3 or 4 days)


### What is in this repository? 

__python__:

`ANKLES.py` : functions for end-to-end inference from un-normalized inputs to un-normalized posteriors given a path to an SBI model.

`priors.py` : the KDEprior object that ANKLES needs to generate posteriors

`demo.ipynb` : demonstration of using `ANKLES.py` to generate posteriors on your new data.

`generate_data_with_COSMIC.ipynb` : exactly what it sounds like. generating the train/test data using a binary population from the `COSMIC` code ([Breivik et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...898...71B/abstract)).

`train_baseline.ipynb` : train a "vanilla" MLP to estimate orbital parameters. It just does OK, and we show the need for probabilistic methods. Resulting MLP is stored in `models/`.

`train_SBI.ipynb` : Where I initially trained the SBI. Resulting model is stored in `models/`. 



__etc etc__:

`data/` - stores training/test data. Note, my training data is too large to upload to github.
1. `cosmic_ibt.csv` :  The binary table output by `COSMIC`. Running the cells below the cosmic cell in the `generate_data_with_COSMIC` notebook, you can generate your own train/test files 
2. `normed_labels.npz` : just the labels of the train/test data. These are smoothed into a KDEprior for SBI to use in sampling/training.


`models/` - store trained models:
1. `baseline_MLP.pth` : a vanilla multilayer perceptron 
2. `sbi_model.pt` : a test SBI model that trained for 1000 epochs on 5e4 data points and did not converge
3. `sbi_model_longtraining.pt` : the "final" SBI Model that trained for 1127 epochs and *did* converge.
4. `training_extrema_sbi.npz` : the min/max values of model parameters used to train the sbi models. these are called in ANKLES.py to un-normalize outputs.


`plots/` - plots for astronomy 205 poster/paper