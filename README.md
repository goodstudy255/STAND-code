# T2Diff
---
## Paper code and data

This is the code for the WWW 2025 underreview paper: [Unleashing the Potential of Two-Tower Models: Diffusion-Based Cross-Interaction for Large-Scale Matching](Paper 296). We have implemented our methods in **Tensorflow**.


These are two datasets we used in our paper， Kuairand and ml-1m respectively. After download them in the link below, you can correct the path into your own in `kuairand_proess.py` and `ml_1m_process.py` respectively.

kuairand dataset: https://kuairand.com/

movielens dataset: https://grouplens.org/datasets/movielens/

Due to varying tag counts associated with each data entry in the 'KuaiRand' and ML-1M datasets, which reflect their unique data acquisition methods, we have crafted a dedicated model file for each. The model path is `./model`.

---

## Usage
The path of configuration file is `config`, you can modify the model path in `model.conf` and modify the relevant parmaters in `nn_param.conf`

As our model is in two different modes at training and testing phase, so you need to run `cmain_multi_graph.py` to train the model.

For example: `python3 cmain_multi_graph.py --model t2diff_ml_1m`


---
## Requirements

. Python 3.7 \
. Tensorflow 1.13.1 \
. tensorboar 1.13.1 \
. scipy 1.7.3 

---



