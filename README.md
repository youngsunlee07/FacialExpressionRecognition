# Facial Expression Recognition (FER) with Deep Learning

This project focuses on comparing and evaluating various deep learning models for facial emotion recognition. It includes models created based on research papers, specifically the DACL and TransFER models, as well as standard models such as AlexNet, ResNet50, and VGGNet, adapted for the FER-2013 dataset. 

## Project Structure 
```bash
FERProject/
│
├── data/
│ ├── train/
│ │ ├── angry/
│ │ ├── disgust/
│ │ ├── ...
│ └── test/
│ ├── angry/
│ ├── disgust/
│ ├── ...
│
├── Alexnet.ipynb
├── DACL.ipynb
├── ResNet-50.ipynb
├── TransFER.ipynb
├── VGGNet.ipynb
└── VGGNet.ipynb
```

## Dataset
The FER-2013 dataset is used, containing images categorized into seven emotions: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.

## Installation
Clone the project and install the required packages.
```bash
git clone https://github.com/youngsunlee07/FERProject.git
cd FERProject
pip install -r requirements.txt
``` 

## Models
- DACL (ResNet18 + Attention)
The DACL model is based on ResNet18 with an additional Attention module. It is implemented in DACL.ipynb.

- TransFER (ResNet18+Transformer) 
The TransFER model combines ResNet18 and a Transformer Encoder. It is implemented in TransFER.ipynb.

- AlexNet, ResNet50, VGGNet
These standard models are adapted for the FER-2013 dataset and are implemented in Alexnet.ipynb, ResNet-50.ipynb, and VGGNet.ipynb, respectively.

## Training
To train a model, open and run the respective Jupyter Notebook file. For example, to train the DACL model, open DACL.ipynb.

## Evaluation
To evaluate a model, run the evaluation section in the respective Jupyter Notebook file. This will display metrics such as Confusion Matrix, Precision, Recall, F1 Score, ROC Curve, and AUC.

## Results
The following metrics are used to measure the performance of the models:

- Confusion Matrix
- Precision
- Recall
- F1 Score
- ROC Curve and AUC

## Contact
For any questions or inquiries, please contact: youngsun.lee07@gmail.com
