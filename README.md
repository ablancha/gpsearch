# gpsearch

Source code for Bayesian optimization and active learning with likelihood-weighted acquisition functions. 

## Installation

Execute `pip install .` from the master directory.

## Notes

Beware of [this issue](https://github.com/SheffieldML/GPy/issues/802) if you are using the `devel` version of `GPy`.  The acquisition functions available in `gpsearch` were implemented before this issue was fixed.

## References

* [Bayesian Optimization with Output-Weighted Importance Sampling](https://arxiv.org/abs/2004.10599)
* [Informative Path Planning for Anomaly Detection in Environment Exploration and Monitoring](https://arxiv.org/abs/2005.10040)
* [Output-Weighted Importance Sampling for Bayesian Experimental Design and Uncertainty Quantification](https://arxiv.org/abs/2006.12394)
