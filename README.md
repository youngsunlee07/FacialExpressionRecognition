# Facial Expression Recognition (FER) with Deep Learning
This project focuses on comparing and evaluating various deep learning models for facial emotion recognition. It includes models created based on research papers, specifically the DACL and TransFER models, as well as standard models such as AlexNet, ResNet50, and VGGNet, adapted for the FER-2013 dataset. 

## Dataset
The FER-2013 dataset is used, containing images categorized into seven emotions: angry, disgust, fear, happy, neutral, sad, and surprise. 

## Usage
### Download the FER-2013 Dataset
You can download the FER-2013 dataset from [Kaggle]
After downloading, extract the files and organize them as follows:

FERProject/
│
├── data/
│ ├── train/
│ │ ├── angry/
│ │ ├── disgust/
│ │ ├── fear/
│ │ ├── happy/
│ │ ├── neutral/
│ │ ├── sad/
│ │ ├── surprise/
│ ├── test/
│ │ ├── angry/
│ │ ├── disgust/
│ │ ├── fear/
│ │ ├── happy/
│ │ ├── neutral/
│ │ ├── sad/
│ │ ├── surprise/ 

### Installation
Clone the project and install the required packages.
```bash
git clone https://github.com/youngsunlee07/FacialExpressionRecognition.git
cd FacialExpressionRecognition
pip install -r requirements.txt
``` 

### Training 
To train a model, run the respective Python file. For example, to train the DACL model, use the following command: 
```bash 
python dacl.py 
```
Before running the command, ensure that the load_data function in each script points to the correct dataset path: 
```bash 
train_loader, test_loader = load_data(r'path/to/train', r'path/to/test', batch_size=32)
```
Replace 'path/to/train' and 'path/to/test' with the actual paths to the dataset.

### Evaluation
To evaluate a model, the script will automatically print evaluation metrics and generate plots after training. The metrics include:
- Confusion Matrix
- Precision
- Recall
- F1 Score
- ROC Curve and AUC

### Results
The following metrics are used to measure the performance of the models:
- Confusion Matrix
- Precision
- Recall
- F1 Score
- ROC Curve and AUC

## Models
### DACL (ResNet18 + Attention) 
The DACL model is based on ResNet18 with an additional Attention module. It is implemented in DACL.ipynb.

### TransFER (ResNet18+Transformer) 
The TransFER model combines ResNet18 and a Transformer Encoder. It is implemented in TransFER.ipynb.

### AlexNet, ResNet50, VGGNet 
These standard models are adapted for the FER-2013 dataset and are implemented in Alexnet.ipynb, ResNet-50.ipynb, and VGGNet.ipynb, respectively.



## Contact
For any questions or inquiries, please contact: youngsun.lee07@gmail.com
