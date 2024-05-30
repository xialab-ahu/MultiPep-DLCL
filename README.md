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
| train.py          | train model                                            |
| models            | model construction                                     |
| DataLoad.py       | data reading and encoding                              |
| loss_functions.py | loss functions used to train models                    |
| evaluation.py     | evaluation metrics (for evaluating prediction results) |
| dataset           | data                                                   |
| result            | Models and results preserved during training.          |
| config            | Model parameters                                       |

## Requirements

The environment is based on:<br />
`pytorch=1.12.1=py3.9_cuda10.2`<br />
`numpy ==1.26.2`<br />
`pandas==1.2.4`.<br />


## Contact
Please feel free to contact us if you need any help.

