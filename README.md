# An end-to-end heterogeneous graph representation learning-based framework for drug-target interaction prediction 

**This repo is moved to [23AIBox](https://github.com/23AIBox/23AIBox-EEG-DTI)**

Accurately identifying potential drug-target interactions (DTIs) is a key step in drug discovery. Although many related experimental studies have been carried out for identifying DTIs in the past few decades, the biological experiment- based DTI identification is still time-consuming and expensive. Therefore, it is of great significance to develop effective computational methods for identifying DTIs. In this paper, we develop a novel end-to-end learning-based framework for drug-target interaction prediction based on heterogeneous graph convolutional networks called EEG-DTI. Given a heterogeneous network containing multiple types of biological entities (i.e. drug, protein, disease, side-effect), EEG-DTI learns the low dimensional feature representation of drugs and targets using a graph convolutional networks-based model and predicts drug-target interactions based on the learned features. During the training process, EEG-DTI learns the feature representation of nodes in an end-to-end mode. The evaluation test shows that EEG-DTI performs better than existing state-of-art methods.


## The environment of EEG-DTI
    Linux OS
    python 3.7.8 
    tensorflow                1.15.0 

## Run the EEG-DTI model for DTI prediction
### Download the data.
The expermental data can be found in this [link](https://github.com/MedicineBiology-AI/EEG-DTI/tree/master).
### Luo dataset 

    $ python main_luo_all_networks.py

### Yamanishi dataset 

    $ python main_yamanashi.py


## Acknowledgments
1. We really thank the SNAP Group open the source code of Decagon at this [link](https://github.com/mims-harvard/decagon). The Decagon help us to finish message passing of nodes on heterogeneous network.

2. We really thank Yunan Luo et al. open the dataset in this papaer "Yunan Luo, Xinbin Zhao, Jingtian Zhou, Jinglin Yang, Yanqing Zhang, Wenhua Kuang, Jian Peng, Ligong Chen, and Jianyang Zeng. A network integration approach for drug-target interaction prediction and computational drug repositioning from heterogeneous information. Nature communications, 8(1):1–13, 2017."

3. We really thank Yoshihiro Yamanishi et al. open the dataset in this papaer "Yoshihiro Yamanishi, Michihiro Araki, Alex Gutteridge, Wataru Honda, and Minoru Kanehisa. Prediction of drug– target interaction networks from the integration of chemical and genomic spaces. Bioinformatics, 24(13):i232–i240, 2008."
