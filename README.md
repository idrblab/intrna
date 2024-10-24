[![AUR](https://img.shields.io/badge/license-GPL%203.0-blue.svg)]
[![Python 3.x](https://img.shields.io/badge/Python-3.X-green.svg)](https://www.python.org/)
[![IntRNA](fig/RNAInterpret-Title.png)](http://idrblab.org/rnamap/)
IntRNA first introduced a dual-path classification model to enable a systematic interpretation of the RNA universe. **(1) providing a dual-path classification model was proposed to enable systematic interpretation of the RNA universe, (2) a novel encoding strategy was introduced by incorporating the most comprehensive set of features**. Therefore, users can get the encoding results and the classification of RNA from IntRNA.

# Contents
- [Contents](#contents)
- [Requirements and installment](#requirements-and-installment)
- [Basic usage](#basic-usage)
- [Study demo](#study-demo)
  - [IntRNA](#1-IntRNA)
- [Question && issue](#question--issue)

# Requirements and installment
This software is developed with Python 3.X, Python 3.X is required as runtime environment. we recommemd user to install [anaconda3](https://www.anaconda.com/) which is well-known as a scientific Python.

```shell
git clone https://github.com/idrblab/intrna
cd intrna
# virtual environment are recommended
conda create -n intrna python=3.8
pip install -r requirements.txt
```
# Basic usage
Users can use IntRNA to **(1) encode RNA image** and **(2) classification of RNA**.
```shell
python IntRNA.py --help
```
```
usage: IntRNA.py [-h] [-i INPUTFASTA] [-o OUTPUTPATH] [-m {1,2,3}]

Your program description here

optional arguments:
  -h, --help            show this help message and exit
  -i INPUTFASTA, --inputfasta INPUTFASTA
                        The RNA data needed to predict
  -o OUTPUTPATH, --outputpath OUTPUTPATH
                        The folder to save encoding features and predict result
  -m {1,2,3}, --model_type {1,2,3}
                        The model will be used to predict. 1: mRNA and ncRNA; 2: 13 classes linear ncRNA; 3: circRNA and linear RNA
```



# Study demo
## IntRNA
Users can run tasks like following shell script.
```shell
python IntRNA.py -i ./demo/sense_intronic_sample.fa \
    -o ./demo/output_sense_intronic_sample \
    -m 1 
```

# Question && issue
If there are any usage problems, welcome to submit issues to this repository or contact author with email lfwyx@zju.edu.cn
