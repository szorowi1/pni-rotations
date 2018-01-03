# moodRL
Reproduction and extension of the analyses in [Eldar & Niv (2015)](https://www.nature.com/articles/ncomms7149). Completed as part of Fall 2017 PNI lab rotations. 

## TODO
**Major Importance**
- [ ] Annotate all plots / code sections.
- [ ] Fit basic RL models to simulated data. Decide on priors.
- [ ] Fit hierachical RL model to simulated data.

**Medium Items**
- [ ] Finish decoding original analysis scripts.
- [ ] Timeseries model linking reward history / RPEs to mood.

**Small Importance**
- [ ] Update docstrings 
- [ ] Merge *optimal_choice* functions across scripts

## References
- [Eldar & Niv (2015)](https://www.nature.com/articles/ncomms7149): original article
- [Daw (2009)](http://www.cns.nyu.edu/~daw/d10.pdf): model fitting
- [Gershman (2016)](http://www.sciencedirect.com/science/article/pii/S0022249616000080): model fitting
- [hBayesDM software](https://github.com/CCS-Lab/hBayesDM): model fitting, Stan code
- [fitr software](https://github.com/abrahamnunes/fitr): model fitting, stan code
- [Gelman (2000)](http://www.stat.columbia.edu/~gelman/research/published/dogs.pdf): model checking (posterior predictive checks) 