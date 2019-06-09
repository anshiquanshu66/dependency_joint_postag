# dependency_joint_postag

This repository includes the code of Joint training Postag and Deep Biaffine Attention for Neural Dependency Parsing.
The implementation is based on the dependency parser by Ma et al. (2018) (https://github.com/XuezheMax/NeuroNLP2) and reuses part of its code.

### Running Environment

This implementation requires Python 3.6, PyTorch 0.4.1 and Gensim >= 0.12.0.
Our suggestion is to use conda to install the required environment:

* `conda create -n myenv python=3.6; source activate myenv; conda install gensim;`
* `conda install pytorch=0.4.1 cudatoolkit=9.2 -c pytorch`

### Running the experiments

In the root of the repository, first make the tmp directory:

    mkdir tmp

To train parser, simply run

    go_train.sh

