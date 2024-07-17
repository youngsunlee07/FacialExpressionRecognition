# Facial Expression Recognition (FER) with Deep Learning
This project focuses on comparing and evaluating various deep learning models for facial emotion recognition. It includes models created based on research papers, specifically the DACL and TransFER models, as well as standard models such as AlexNet, ResNet50, and VGGNet, adapted for the FER-2013 dataset. 

## Dataset
The FER-2013 dataset is used, containing images categorized into seven emotions: angry, disgust, fear, happy, neutral, sad, and surprise. 

## Usage
### Download the FER-2013 Dataset
You can download the FER-2013 dataset from [Kaggle]
After downloading, extract the files and organize them as follows:

```perl
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
```

### Installation
Clone the project and install the required packages.
```bash
git clone https://github.com/youngsunlee07/FacialExpressionRecognition.git
cd FacialExpressionRecognition
pip install -r requirements.txt
``` 

### Running on Google Colab
To run this project on Google Colab, follow these steps: 

1. Upload Your Files: Create a folder named FacialExpressionRecognition in your Google Drive and upload all the .py files (alexnet.py, common.py, dacl.py, resnet-50.py, transfer.py, vggnet.py) and the FER-2013 directory to it. 

2. Set Runtime Type: In Google Colab, change the runtime type to use a GPU:
- Click on Runtime in the top menu.
- Select Change runtime type.
- Set Hardware accelerator to GPU.
- Click Save.

3. Mount Google Drive: Mount your Google Drive in the Colab notebook. 
```bash 
from google.colab import drive
drive.mount('/content/drive')
```

4. Navigate to Project Directory: Change the directory to the project folder in Google Drive.
```bash 
import os
os.chdir('/content/drive/MyDrive/FacialExpressionRecognition')
```

5. Install Requirements: Install the required packages (only if you haven't already installed them).
```bash 
!pip install -r requirements.txt 
```

6. Run the Model: Execute the desired model script.
```bash 
!python alexnet.py 
```

## Models
### DACL (ResNet18 + Attention) 
The DACL model is based on ResNet18 with an additional Attention module. It is implemented in dacl.py

### TransFER (ResNet18+Transformer) 
The TransFER model combines ResNet18 and a Transformer Encoder. It is implemented in transfer.py

### AlexNet, ResNet50, VGGNet 
These standard models are adapted for the FER-2013 dataset and are implemented in alexnet.py, resnet-50.py, and vggnet.py respectively.

## Contact
For any questions or inquiries, please contact: youngsun.lee07@gmail.com 