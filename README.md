# PAD-UFES20 Skin Lesion Classification

## Overview

This project investigates binary skin lesion classification using the PAD-UFES-20 dataset, which contains smartphone-acquired clinical images and patient clinical features. The aim was to compare traditional machine learning models trained on tabular clinical features with deep learning models trained on image data.

The project was completed as part of the Data Mining and Machine Learning coursework at Heriot-Watt University.

## Dataset

The PAD-UFES-20 dataset contains clinical images and patient information collected through a real-world teledermatology system in Brazil. The dataset includes smartphone-acquired skin lesion images and clinical variables such as age, lesion location, skin type, and lesion diameter.

For this project, lesion diagnoses were grouped into binary classes:

* Cancerous: BCC, MEL, SCC
* Benign: ACK, NEV, SEK

The dataset is publicly available on Mendeley Data under a CC BY 4.0 licence.

## Research Questions

* Can smartphone-acquired clinical images support reliable binary skin lesion classification?
* How do traditional machine learning models trained on clinical tabular features compare with deep learning models trained on image data?
* Do patient clinical features provide more stable predictive information than image-only models under variable smartphone image acquisition conditions?

## Methods

### Data Preprocessing

* Removed irrelevant identifiers such as patient and lesion IDs.
* Handled missing values.
* One-hot encoded categorical clinical variables.
* Scaled numerical features.
* Resized image data to 200 × 200 pixels.
* Normalised RGB image values.
* Matched image samples with their corresponding clinical labels.

### Classical Machine Learning Models

The following models were trained on tabular clinical features:

* Logistic Regression
* Naive Bayes
* Decision Tree
* k-Nearest Neighbours

### Deep Learning Models

The following deep learning models were implemented:

* Multi-Layer Perceptron (MLP) trained on tabular clinical features
* MLP trained on image data
* Convolutional Neural Network (CNN)
* ResNet-based CNN experiments

## Evaluation

Model performance was assessed using clinically relevant classification metrics, including:

* Accuracy
* Precision
* Recall
* F1-score
* Sensitivity
* Specificity
* ROC-AUC
* Matthews Correlation Coefficient (MCC)
* Confusion matrices

These metrics were selected because the dataset showed class imbalance, with malignant cases forming the majority class.

## Results Summary

| Model                   | Accuracy | F1-score | Sensitivity | Specificity |  MCC |
| ----------------------- | -------: | -------: | ----------: | ----------: | ---: |
| Logistic Regression     |      86% |     0.89 |        0.89 |        0.81 | 0.63 |
| MLP on Tabular Features |      83% |     0.88 |        0.89 |        0.72 | 0.62 |
| Decision Tree           |      80% |     0.84 |        0.86 |        0.71 | 0.57 |
| k-Nearest Neighbours    |      80% |     0.85 |        0.92 |        0.61 | 0.58 |
| Naive Bayes             |      79% |     0.86 |        0.95 |        0.46 | 0.50 |
| CNN                     |      66% |     0.71 |        0.61 |        0.76 | 0.35 |
| MLP on Image Data       |      58% |     0.63 |        0.53 |        0.70 | 0.21 |

## Key Findings

* Tabular clinical features outperformed image-only deep learning models.
* Logistic Regression achieved the strongest overall performance.
* The MLP trained on tabular clinical features also performed strongly.
* Image-based models struggled with variability in smartphone image quality, including differences in lighting, focus, colour saturation, lesion size, and artefacts.
* Clinical symptoms such as bleeding, growth, elevation, itching, and pain showed stronger correlation with malignancy than image-only features.

## Repository Structure

* `notebooks/` – preprocessing, exploratory analysis, modelling, and evaluation notebooks
* `notebooks/not_used/` – exploratory model experiments not included in the final evaluation
* `scripts/` – dataset utility scripts
* `models/` – trained model weights and preprocessing scaler
* `documentation/` – project presentation
* `data/` – proposal document and example dataset files
  
## My Contribution

- Clinical data preprocessing and feature engineering
- Image preprocessing and preparation of deep learning pipelines
- Development, training, and evaluation of CNN architectures (Custom CNN and ResNet-18)
- Dataset sourcing, organisation, and preparation
- Preparation of project presentation materials and report templates
- Co-authoring the Dataset Description and Experimental Setup sections of the final report
  
## Contributors

This project was completed as a group MSc coursework project.

Contributors:

* Mercy Nthiwa
* Christa Mary Veluthedath Jacob
* Gonzalo Larroya
* Oriol Linan
* Vedhagiri Alagesan

Repository maintained by Mercy Nthiwa for portfolio and educational purposes.

## Limitations

* The dataset was relatively small and imbalanced.
* Diagnoses were consensus-based rather than biopsy-confirmed.
* Smartphone image quality varied substantially.
* No external validation dataset was used.
* Missing metadata reduced the reliability of some tabular features.

## Future Work

* Explore multimodal fusion combining image and clinical features.
* Evaluate transfer learning using pretrained CNN and vision foundation models.
* Investigate missing-data-robust architectures.
* Apply explainability methods such as Grad-CAM and SHAP.
* Validate the models on external datasets and more diverse skin tones.

## Licence and Data Attribution

The PAD-UFES-20 dataset is publicly available on Mendeley Data under the CC BY 4.0 licence. Dataset users should cite the original dataset publication and follow the licence terms.

This repository is provided for educational and portfolio purposes.

## Acknowledgements

This work was completed as part of a group MSc coursework project at Heriot-Watt University. I would like to acknowledge my project teammates for their contributions to the overall coursework submission.

