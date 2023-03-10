# gpsearch

Source code for Bayesian optimization and active learning with likelihood-weighted acquisition functions. 

## Installation

Clone the repo, then create a fresh conda environment from the requirements file and install using `pip`.

```
git clone https://github.com/ablancha/gpsearch.git
cd gpsearch
conda create --name myenv --file requirements.txt -c conda-forge -c dmentipl
conda activate myenv
pip install .
```

## Notes
The acquisition functions available in `gpsearch` are compatible with 1.9.9 of GPy. Beware of [this issue](https://github.com/SheffieldML/GPy/issues/802) if you decide to use a different version.

## Benchmarks

The following benchmarks are included:
* stochastic oscillator (used [here](https://arxiv.org/abs/2006.12394))
* extreme-event precursor (used [here](https://arxiv.org/abs/2004.10599) and [here](https://arxiv.org/abs/2006.12394))
* borehole function (used [here](https://arxiv.org/abs/2006.12394))
* synthetic test functions (used [here](https://arxiv.org/abs/2004.10599))
* brachistochrone problem (unpublished)

## References

* [Bayesian Optimization with Output-Weighted Optimal Sampling](https://arxiv.org/abs/2004.10599)
* [Informative Path Planning for Extreme Anomaly Detection in Environment Exploration and Monitoring](https://arxiv.org/abs/2005.10040)
* [Output-Weighted Optimal Sampling for Bayesian Experimental Design and Uncertainty Quantification](https://arxiv.org/abs/2006.12394)

