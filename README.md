# STAND
---
## Paper code and data
https://github.com/goodstudy255/STAND-code/blob/main/README.md
This is the code for the Recsys 2024 underreview paper: [Unleashing the Potential of Two-Twoer Models: Diffusion-Based Cross-Interaction for Large-Scale Matching](Paper 161). We have implemented our methods in **Tensorflow**.

These are two datasets we used in our paper， Kuairand and ml-1m respectively. After download them, you can correct the path into your own in `dataset_proess.py` and `ml_1m_process.py` respectively.


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



