# 🫁 Multimodal Deep Learning for Pneumonia Detection

A deep learning project for automated **pneumonia detection from chest X-ray images** using the **RSNA Pneumonia Detection Challenge Dataset**.

The project investigates whether combining chest X-ray images with patient demographic metadata — **age, sex, and X-ray view position** — can improve pneumonia classification compared with traditional image-only deep learning approaches.

Three models are developed and compared:

1. **Custom Convolutional Neural Network (CNN)**
2. **EfficientNetB0 Image-Only Model**
3. **Multimodal EfficientNetB0 Model (Image + Patient Metadata)**

The multimodal model combines features extracted from chest X-ray images with structured patient information before performing binary classification.

---

## 📌 Project Overview

Deep learning has demonstrated strong performance in medical image analysis, particularly for chest X-ray classification. However, many pneumonia detection systems rely only on image information and do not incorporate patient-specific information that may be relevant during clinical diagnosis.

This project explores a **multimodal deep learning approach** that combines:

* Chest X-ray images
* Patient age
* Patient sex
* X-ray view position

The aim is to determine whether integrating demographic metadata with radiographic image features can improve automated pneumonia detection.

---

## 🎯 Objectives

The main objectives of this project are:

* Develop a deep learning framework for pneumonia detection from chest X-ray images.
* Extract and preprocess patient demographic metadata from DICOM files.
* Develop a baseline Custom CNN model.
* Develop an EfficientNetB0 transfer-learning model.
* Develop a multimodal EfficientNetB0 model combining image and metadata features.
* Perform hyperparameter tuning to improve model performance.
* Compare image-only and multimodal approaches.
* Evaluate the models using several medical image classification metrics.

---

## 📊 Dataset

This project uses the **RSNA Pneumonia Detection Challenge Dataset**, provided by the Radiological Society of North America (RSNA) through Kaggle.

**Dataset:**
https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/

The dataset contains chest radiographs stored in **DICOM (Digital Imaging and Communications in Medicine)** format.

In addition to the image data, information extracted from the DICOM metadata includes:

* **Patient Age**
* **Patient Sex**
* **View Position**

The task is treated as a binary classification problem:

* `0` — No Pneumonia
* `1` — Pneumonia

> **Note:** The RSNA dataset is not included in this GitHub repository because of its size. Download the dataset directly from Kaggle using the link above.

---

## 🧠 Models

### 1. Custom CNN

A custom Convolutional Neural Network was developed as the baseline image classification model.

The architecture includes:

* Convolutional layers
* Batch normalization
* Max pooling
* Global average pooling
* Dense layers
* Dropout
* Sigmoid output layer

The Custom CNN uses only chest X-ray images.

---

### 2. EfficientNetB0 Image-Only Model

The second model uses **EfficientNetB0** with transfer learning.

ImageNet pretrained weights are used to provide previously learned visual representations.

The EfficientNetB0 backbone extracts features from the chest X-ray images before they are passed through classification layers for pneumonia prediction.

This model also uses only image information.

---

### 3. Multimodal EfficientNetB0

The proposed model combines **chest X-ray images and patient demographic metadata**.

The architecture contains two branches.

#### Image Branch

The image branch uses **EfficientNetB0** to extract visual features from chest X-ray images.

#### Metadata Branch

The metadata branch processes:

* Patient age
* Patient sex
* View position

using fully connected neural network layers.

#### Feature Fusion

The representations produced by the image and metadata branches are concatenated into a combined feature representation.

The fused features are then passed through fully connected layers before a sigmoid output layer performs the final pneumonia classification.

---

## ⚙️ Data Preprocessing

### Image Preprocessing

The DICOM chest X-ray images undergo several preprocessing steps:

* DICOM image loading using `pydicom`
* Pixel intensity processing
* MONOCHROME1 image inversion where required
* Percentile-based intensity clipping
* Image normalization
* Conversion to three channels
* Image resizing to **224 × 224**
* Data augmentation during training

Data augmentation includes transformations such as:

* Random horizontal flipping
* Rotation
* Zoom
* Contrast adjustment

### Metadata Preprocessing

Patient metadata is processed separately.

The preprocessing pipeline includes:

* Missing-value imputation
* Age standardization
* Categorical variable processing
* One-hot encoding

The processed metadata is then supplied to the metadata branch of the multimodal model.

---

## 🔀 Dataset Split

The dataset is divided using stratified sampling to maintain the class distribution:

| Dataset    | Percentage |
| ---------- | ---------: |
| Training   |        75% |
| Validation |        10% |
| Testing    |        15% |

The training set is used to learn model parameters.

The validation set is used for hyperparameter tuning, model selection, threshold selection, and monitoring training performance.

The test set remains independent and is used for final model evaluation.

---

## 🔧 Hyperparameter Tuning

Hyperparameter tuning is performed for the proposed multimodal model.

Several configurations are evaluated using the validation dataset.

Parameters investigated include:

* Learning rate
* Fine-tuning learning rate
* Dense layer size
* Dropout rate
* L2 regularization
* EfficientNetB0 fine-tuning fraction

The best configuration is selected primarily using **validation ROC-AUC**.

The selected configuration is subsequently used to train and fine-tune the final multimodal model.

---

## 📈 Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall / Sensitivity
* Specificity
* F1-Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

Using multiple evaluation metrics provides a more complete assessment of model performance, particularly because pneumonia classification may involve class imbalance.

---

## 📊 Results

The proposed **Multimodal EfficientNetB0** achieved the strongest overall classification performance among the three models.

Key results for the multimodal model include:

| Metric      |     Result |
| ----------- | ---------: |
| Accuracy    | **81.16%** |
| Precision   | **58.75%** |
| Specificity | **88.75%** |
| ROC-AUC     | **0.8454** |

The multimodal model achieved improved overall discrimination and reduced false-positive predictions compared with the baseline models.

The results suggest that combining demographic metadata with chest X-ray image features can provide useful additional information for automated pneumonia classification.

---

## 🛠️ Technologies and Libraries

The project was implemented using **Python** and **Google Colab**.

Major libraries include:

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* pydicom
* scikit-learn
* Matplotlib
* EfficientNetB0

---

## 🚀 Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/JolomiOteri/multimodal-pneumonia-detection.git
```

Move into the project directory:

```bash
cd multimodal-pneumonia-detection
```

### 2. Download the Dataset

Download the RSNA Pneumonia Detection Challenge dataset from Kaggle:

https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/

You may need to accept the competition rules before downloading the dataset.

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install tensorflow pandas numpy pydicom scikit-learn matplotlib
```

### 4. Open the Notebook

The main implementation is available in:

```text
RSNA_Multimodal_Pneumonia_Detection.ipynb
```

The notebook can be opened using:

* Google Colab
* Jupyter Notebook
* JupyterLab
* Visual Studio Code

Google Colab with GPU acceleration is recommended for model training.

---

## 📁 Repository Structure

```text
multimodal-pneumonia-detection/
│
├── README.md
│
├── RSNA_Multimodal_Pneumonia_Detection.ipynb
│
├── requirements.txt
│
├── .gitignore
│
├── results/
│   ├── confusion_matrix_custom_cnn.png
│   ├── confusion_matrix_efficientnet.png
│   ├── confusion_matrix_multimodal.png
│   └── roc_curve_comparison.png
│
└── docs/
    └── dissertation.pdf
```

---

## 🔬 Research Contribution

The main contribution of this project is the development of a **multimodal pneumonia classification framework** that combines radiographic image information with structured patient demographic metadata.

Instead of relying solely on chest X-ray images, the proposed architecture incorporates **age, sex, and view position** alongside features extracted using EfficientNetB0.

This approach investigates how multimodal information can contribute to more clinically relevant AI-assisted medical image classification.

---

## ⚠️ Disclaimer

This project was developed for **academic and research purposes only**.

The models and predictions produced by this project are **not intended for clinical diagnosis or direct patient care**. The system has not undergone clinical validation and should not be used as a substitute for professional medical assessment.

---

## 👨‍💻 Author

**Oritsejolomisan Oteri**

MSc Artificial Intelligence and Data Science
University of East London

---

## 📚 Project Context

This repository contains the implementation developed as part of an MSc research project investigating:

**“Multimodal Deep Learning for Pneumonia Detection: Integrating Chest X-ray Imaging and Clinical Data for Improved Diagnostic Accuracy.”**

The research compares conventional image-only deep learning approaches with a multimodal architecture integrating patient demographic metadata.

---

## ⭐ Acknowledgements

The project uses the **RSNA Pneumonia Detection Challenge Dataset**, made publicly available through RSNA and Kaggle.

TensorFlow, Keras, scikit-learn, pydicom, NumPy, Pandas, and Matplotlib were used for model development, preprocessing, evaluation, and visualisation.
