# T2Diff
---
## Paper code and data

This is the code for the Recsys 2024 underreview paper: [Unleashing the Potential of Two-Tower Models: Diffusion-Based Cross-Interaction for Large-Scale Matching](Paper 161). We have implemented our methods in **Tensorflow**.


These are two datasets we used in our paper， Kuairand and ml-1m respectively. After download them in the link below, you can correct the path into your own in `kuairand_proess.py` and `ml_1m_process.py` respectively.

kuairand dataset: https://kuairand.com/

movielens dataset: https://grouplens.org/datasets/movielens/

Due to data security protocols, the 'KuaiRand' dataset is accessible only upon submission of an email request. Consequently, we have omitted the preprocessed outcomes for 'KuaiRand'. You can find the preprocessed results for the ML-1M dataset at `./ml_1m`.

Due to varying tag counts associated with each data entry in the 'KuaiRand' and ML-1M datasets, which reflect their unique data acquisition methods, we have crafted a dedicated model file for each. The model path is `./model`.

---

## Usage
The path of configuration file is `config`, you can modify the model path in `model.conf` and modify the relevant parmaters in `nn_param.conf`

As our model is in two different modes at training and testing phase, so you need to run `cmain_multi_graph.py` to train our model. As for the other comparison models, you can run the file`cmain.py` to train the model. Both the model use the same configuration file.

For example: `python3 cmain_multi_graph.py` and `python3 cmain.py`


---
## Requirements

. Python 3.7 \
. Tensorflow 1.13.1 \
. tensorboar 1.13.1 \
. scipy 1.7.3 

---



