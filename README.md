
# Generator Reweighting

<center>
<img src="assets/0_0.png" alt= “” width="300">
</center>

<p align="center">
  <img src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</p>

The quadratic equation is given by \[ax^2 + bx + c = 0.\]



A repository for reweighting between different neutrino generators. 

It works by training a BDT to discriminate between the two generators and then using the BDT to create weights for the events in the generator that we want to reweight to. 

The weights are created using the likelihood ratio trick. E.g let $f(x)$ be a classifier trained with the binary cross entropy loss then the likelihood ratio is given by:
$$w_x = \frac{f(x)}{1-f(x)}$$


## Getting Started
To clone this branch only: 
```bash	
git clone --branch sklearn --single-branch https://github.com/radiradev/generator_reweight_bdt
```
On lxplus we can source neccessary packages using an LCG view, e.g:
```
source /cvmfs/sft.cern.ch/lcg/views/LCG_103swan/x86_64-centos7-gcc11-opt/setup.sh
```


On a local environment we can install packages using `mamba`. 


##  Basic Usage 
It contains a top level script `run.sh` that will first create a config file, train a BDT, create weights using the trained BDT and finally make plots using the weights. 

The script takes an argument specifying the flux to use, either `dune` or `flat`. So for instance we can run the script using the dune flux with: 

```bash 
bash run.sh dune
```

The scripts can also be ran individually (e.g to recreate plots or enforce some changes to weights). 

## Usage of individual scripts
### Flux config

The flux config creation scripts must be ran from the top level directory of the repository.
Example: 
```
python3 scripts/dune.py (or python3 scripts/flat.py)
```
The script will create a config file in the `config` directory named `files.ini` which contains the paths to the files. 

### Training
This script will train a BDT using the files specified in the `files.ini` config file and save the trained BDT in the `trained_bdt\{generator_a}_to_{generator_b}` directory. Where `generator_a` and `generator_b` are the names of the generators used to train the BDT specified in the `files.ini` config file.
  
```
python3 train.py
```

As a sidenote, in this step we can use any classifier that can discriminate between the two generators (e.g. a neural network, BDT, Random Forest, etc). The classifier currently used is [Histogram-based BDT](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) from `sklearn`.

### Creating weights
This script will create weights for the files specified in the config corresponding to `generator_a` and will save the weights in the same directory as the trained BDT.

```
python3 reweight.py
```

### Creating plots 
This script will create plots using the weights created in the previous step. The plots will be saved in the `plots\{generator_a}_to_{generator_b}` directory. There is a config file in the `config\plots.py` which contains the settings for the plots. 
``` 
python3 plot.py
```



## Extra information
- [Tutorial](https://hsf-training.github.io/analysis-essentials/advanced-python/45DemoReweighting.html) on BDT reweighting from the HEP Software Foundation.
- LBL [presentation](https://indico.fnal.gov/event/47708/contributions/208129/attachments/139833/175623/cv_generatorrw_20210208.pdf) by Cristovao Vilela, and [the repository](https://github.com/cvilelahep/GeneratorReweight/) that this work is based on.