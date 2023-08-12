# HGRN

Official implementation of Hierarchically Gated Recurrent Neural Network for Sequence Modeling. This repo does not contain specific codes, but only scripts and some instructions on how to reproduce the results of the paper. The overall directory is as follows:



## Overall Architecture

The overall network architecture is as follows:

<div  align="center"> <img src="./hgrn.png" width = "100%" height = "100%" alt="network" align=center /></div>

 

## Experiments

### Environment Preparation

Our experiment uses two conda environments, where Autoregressive language modeling, needs to configure the environment according to the Env1 part, and LRA needs to configure the environment according to the Env2 part.

#### Env1

First build the conda environment based on the yaml file:

```
conda env create --file env1.yaml
```

If you meet an error when installing torch, just remove torch and torchvision in the yaml file, rerun the above command, and then run the below commands:

```
conda activate hgrn
wget https://download.pytorch.org/whl/cu111/torch-1.8.1%2Bcu111-cp36-cp36m-linux_x86_64.whl
pip install torch-1.8.1+cu111-cp36-cp36m-linux_x86_64.whl
pip install -r requirements_hgrn.txt
```

Then install `hgru-pytorch`:
```
conda activate hgrn
cd hgru-pytorch
pip install .
```

Finaly install our version of fairseq:

```
cd fairseq
pip install --editable ./
```



#### Env2

Build the conda environment based on the yaml file:

```
conda env create --file env2.yaml
```



### Autoregressive language model

#### 1) Preprocess the data

First download and prepare the [WikiText-103 dataset](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/):

```
path_to_fairseq=fairseq
cd $path_to_fairseq/examples/language_model/
bash prepare-wikitext-103.sh
cd ../..
```

Next preprocess/binarize the data:

```
TEXT=examples/language_model/wikitext-103
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20
```

This step comes from [fairseq](https://github.com/facebookresearch/fairseq/blob/main/examples/language_model/README.md).



#### 2) Train the autoregressive language model

Use the following command to train language model:

```
bash script_alm.sh
```

You should change data_dir to preprocessed data.



### Image modeling

```
bash script_im.sh
```


### LRA

#### 1) Preparation

Download the codebase:

```
git clone https://github.com/OpenNLPLab/lra.git
```

Download the data:

```
wget https://storage.googleapis.com/long-range-arena/lra_release.gz
mv lra_release.gz lra_release.tar.gz 
tar -xvf lra_release.tar.gz
```


#### 2) Training

Use the following script to run the experiments, you should change `PREFIX` to your lra path, change `tasks` to a specific task and change `model_config` to t1 or t2:

```
python script_lra.py
```



## Standalone code
See hgru-pytorch

