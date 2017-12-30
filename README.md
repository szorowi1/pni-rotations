# moodRL
Reproduction and extension of the analyses in [Eldar & Niv (2015)](https://www.nature.com/articles/ncomms7149). Completed as part of Fall 2017 PNI lab rotations. 

## TODO
**Analysis**
- Compare scripts to originals.
- Fix hierarchical traditional model.
- Timeseries model linking reward history / RPEs to mood.
- Two-learning rate (positive/negative RPE) models

**Plotting**
- Separate optimal choice by machine
- Plot q-values per machine

**Nitty-gritty**
- Merge *optimal_choice* functions across scripts
- Stan recompilation for no-pooling
- Fix plotting naming conventions + datetime
- Update docstrings (e.g. utility scripts)
- Update demos notebook
- Annotate plots

## Questions
- Theoretical: Relation of mood, cumulative reward, and RPE?
- Theoretical: Relation between losses, motivation, and learning rate?
- Technical: Reasonable range of priors?

## References
- [Eldar & Niv (2015)](https://www.nature.com/articles/ncomms7149): original article
- [Daw (2009)](http://www.cns.nyu.edu/~daw/d10.pdf): model fitting
- [Gershman (2016)](http://www.sciencedirect.com/science/article/pii/S0022249616000080): model fitting
- [hBayesDM software](https://github.com/CCS-Lab/hBayesDM): model fitting, Stan code
- [fitr software](https://github.com/abrahamnunes/fitr): model fitting, stan code
- [Gelman (2000)](http://www.stat.columbia.edu/~gelman/research/published/dogs.pdf): model checking (posterior predictive checks) 