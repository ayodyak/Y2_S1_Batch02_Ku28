# Y2_S1_Batch02_Ku28

Plant Disease Detection using AI

# Overview

This project focuses on building an AI-based Plant Disease Detection Model. The work is divided among 6 group members, where each applies one image preprocessing technique.
The dataset is first downloaded from Kaggle by one member, then pushed to GitHub. After that, each subsequent member clones the repository, applies their preprocessing step in their own notebook, and pushes the updated dataset back.

Finally, the dataset will be used to train and evaluate deep learning models for plant disease detection.
Dataset Details
- Source: Plant Disease Detection Dataset (Kaggle-https://www.kaggle.com/datasets/karagwaanntreasure/plant-disease-detection)
- Type: Images of healthy and diseased plant leaves
- Format: .jpg / .png images
- Classes: Multiple plant species & disease categories

# Group Members & Roles

IT24103933	- Resizing Images	
IT24104378	-	Normalization (Scale Retinex)	
IT24103341	-	Data Augmentation	
IT24103063	-	Color Space Conversion	
it24102867	-	Noise Reduction	
IT24102503	-	Segmentation	

# Project Workflow

1. Download Dataset (Step 0)
   - Use Kaggle API token to download dataset in Colab.
   - Unzip dataset and push it to GitHub.

2. Preprocessing Steps (Step 1 → Step 6)
   - Each member clones the latest repo.
   - Runs their assigned notebook to apply the preprocessing technique.
   - Pushes the updated dataset back to GitHub.

3. Final Dataset
   - After Step 6, the dataset is ready for model training.

# Setup Instructions

1. Clone Repository

!git clone https://github.com/<username>/<repo_name>.git

2. Download Dataset (Only Step 0 – First Member)

!pip install kaggle

from google.colab import files
files.upload()   # upload kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d karagwaanntreasure/plant-disease-detection
!unzip plant-disease-detection.zip -d dataset/

3. Apply Preprocessing 

Open your assigned notebook (.ipynb) and run it.
Example: Resizing (IT24103933)
import cv2, os
input_folder = 'dataset/raw/'
output_folder = 'dataset/raw/technic 1'
os.makedirs(output_folder, exist_ok=True)

for img_name in os.listdir(input_folder):
    img = cv2.imread(os.path.join(input_folder, img_name))
    resized = cv2.resize(img, (224, 224))
    cv2.imwrite(os.path.join(output_folder, img_name), resized)

4. Push Changes to GitHub

!git add .
!git commit -m 'Step 1: technic 1 applied'
!git push origin main
 
# Tools & Libraries
- Google Colab
- Python
- OpenCV – image processing
- TensorFlow / PyTorch – model training
- NumPy, Matplotlib – visualization
- Kaggle API – dataset download
