# MutiPep-DLCL
MultiPep-DLCL:Recognition of Multi-functional Therapeutic Peptides through Deep Learning with Label-Sequence Contrastive Learning


## Introduction
In this work, the MultiPep-DLCL model is proposed to predict MFTP. This work has the following advantages over existing methods:  
(1) Learning sequence features from both global and local features.<br />
(2) Introducing label semantic information and using Transformer to establish the correlation between local features of sequences, global features and label embeddings, and obtaining sequence-related label embeddings.<br />(3) By introducing convenient label-sequence contrastive learning to further guide the related sequence and label feature expressions closer together.<br />
(4) Combining MLFDL, which deals with the dataset imbalance problem, with CEL to train the model, resulting in further improvement of the model performance.<br />

## Related Files

#### MutiPep-DLCL

| FILE NAME         | DESCRIPTION                                            |
|:------------------|:-------------------------------------------------------|
| main.py           | the main file of MutiPep-DLCL recognizer               |
| train.py          | train and predict model                                |
| models            | model construction                                     |
| DataLoad.py       | data reading and encoding                              |
| loss_functions.py | loss functions used to train models                    |
| evaluation.py     | evaluation metrics (for evaluating prediction results) |
| dataset           | data:text.txt is for test set,train.txt is train set   |
| result            | Models and results preserved during training.          |
| config            | Some of the defined model parameters                                       |


## Installation


- Requirements<br />
OS：
  
  - `Windows` ：Windows10 or later
  
  - `Linux`：Ubuntu 16.04 LTS or later<br />

  python: 
  
   Our code runs in the following corresponding versions of the python library, please make sure your environment is compatible with our version: <br />
  `pytorch=1.12.1=py3.9.16_cuda11.6`<br />
  `numpy ==1.26.2`<br />
  `pandas==1.2.4`<br />
- Download `MutiPep-DLCL`to your computer
  ```bash
  git clone https://github.com/xialab-ahu/MultiPep-DLCL
  ```
## Training and test MutiPep-DLCL model
```shell
cd "./MutiPep-DLCL"
python main.py
```

## Contact
Please feel free to contact us if you need any help.

