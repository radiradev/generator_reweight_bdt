
# Generator Reweighting

<center>
<img src="assets/0_0.png" alt= “” width="300">
</center>

A repository for reweighting between different neutrino generators. 

It works by training a BDT to discriminate between the two generators and then using the BDT to create weights for the events in the generator that we want to reweight to. 

The weights are created using the likelihood ratio trick. E.g let $f(x)$ be a classifier trained with the binary cross entropy loss then the likelihood ratio is given by:
$$w_x = \frac{f(x)}{1-f(x)}$$

## To do 

- [ ] interface model with nusystematics by creating a reweight engine - it should take 6 variables from an event `[Enu_true, CosLep, ELep, EavAlt, Erec_bias_abs, Erec_bias_rel]` and would produce a weight.
- [ ] investigate why particle multiplicities do not work with all FSI models, including ways of keeping `Enu_true, CosLep, and ELep` fixed while reweighting the other variables.



## Getting Started
To clone this branch only: 
```bash	
git clone https://github.com/radiradev/generator_reweight_bdt
```

## Environment setup

One should probably setup their own environment so they have full control, however as a quick way to run the code you can use this conda environment:
```
conda activate /afs/cern.ch/work/r/rradev/public/envs/pyg_gpu
``` 
### LCG Views 
- [ ] Add instructions

~~On lxplus we can source neccessary packages using an LCG view, however this uses an old version of `scikit-learn` without `HistGradientBoostingClassifier`:~~
```
source /cvmfs/sft.cern.ch/lcg/views/LCG_103swan/x86_64-centos7-gcc11-opt/setup.sh
```


##  Basic Usage 
It contains a top level script `run.sh` that accepts a `.yaml` file specied in `/config` it specifies the variables to reweight, the files for the nominal and target model and directories that would be used to save the files. The script first trains a BDT, then creates weights associated to the nominal file(s) and finally makes plots comparing the nominal, target and reweighted distributions.
As an example:

```bash 
bash run.sh hA_10a_to_hN2018.yaml
```

The scripts can also be ran individually (e.g to recreate plots or enforce some changes to weights). 

## Usage of individual scripts
You can also run the files indvidually, again you have to provide the config file for each python script.
### Training
This script will train a BDT using the files specified in the `.yaml` config file and save the trained BDT in the `trained_bdt\{nominal_name}_to_{target_name}` directory. Where `nominal_name` and `target_name` are the names of the generators used to train the BDT specified in the `.yaml` config file.
  
```
python3 train.py hA_10a_to_hN2018.yaml
```

As a sidenote, in this step we can use any classifier that can discriminate between the two generators (e.g. a neural network, BDT, Random Forest, etc). The classifier currently used is [Histogram-based BDT](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) from `sklearn`.

### Creating weights
This script will create weights for the files specified in the config corresponding to `generator_a` and will save the weights in the same directory as the trained BDT.

```
python3 reweight.py hA_10a_to_hN2018.yaml
```

### Creating plots 
This script will create plots using the weights created in the previous step. The plots will be saved in the `plots\{nominal_name}_to_{target_name}` directory. There is a config file in the `config\plots.yaml` which contains the settings for the plots. 
``` 
python3 plot.py hA_10a_to_hN2018.yaml
```



## Extra information
- [Tutorial](https://hsf-training.github.io/analysis-essentials/advanced-python/45DemoReweighting.html) on BDT reweighting from the HEP Software Foundation.
- LBL [presentation](https://indico.fnal.gov/event/47708/contributions/208129/attachments/139833/175623/cv_generatorrw_20210208.pdf) by Cristovao Vilela, and [the repository](https://github.com/cvilelahep/GeneratorReweight/) that this work is based on.
