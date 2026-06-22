# Possible Datasets

This document summarizes datasets considered for this project.

## Brain Stroke

### Numerical

- [patient-information](https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-dataset/data)

| gender | age | hypertension | heart_disease | ever_married | work_type | Residence_type | avg_glucose_level |  bmi  | smoking_status   | stroke |
|--------|-----|--------------|---------------|--------------|-----------|----------------|-------------------|-------|------------------|--------|
| Male   | 67  | 0            | 1             | Yes          | Private   | Urban          | 228.69            | 36.6  | formerly smoked  | 1      |
| Male   | 80  | 0            | 1             | Yes          | Private   | Rural          | 105.92            | 32.5  | never smoked     | 1      |
| Female | 49  | 0            | 0             | Yes          | Private   | Urban          | 171.23            | 34.4  | smokes           | 1      |

- [patient-information](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8641997/?utm_source=chatgpt.com#B16)

### Image

- [ct-scan](https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset/data)

![image](https://storage.googleapis.com/kagglesdsdata/datasets/6925704/11154087/Brain_Stroke_CT_Dataset/Bleeding/PNG/10002.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250915%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250915T191436Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=65e88e0f9455c61956b6413a371675e98c323977d6e52f9e9e80facbbf3cdce57081543518637c49d01d5c9ab6266b26297d8fde9205f2414f8150eb144fc6bb3b19734f531f6e7f06bb3c5b78a5e675882a929011544e64337d7d8ef25acd0bc635a752ce020a200429aa7c557f62ec33a1eb10e16765cba12c67361a35a1dc93baab2d6b06a1ba5bf12f5a146060abc9e0c6ba1df5542199de36e7ee39ae8bec54362651c5b84e01dc10685e31c1a862c2efb53ba9f25bcef82925fa14427b77cb2811a6aee33faa3072a19bdbb8323494730907323d2a567bdcecd03ac3d5f72bf620cac87303076e633cb00078d792850811cd14b17e2c79d48374989e02)

## Knee Osteoarthritis

### Numerical

- [patient-information](https://datasetsearch.research.google.com/search?src=0&query=Osteoarthritis&docid=L2cvMTFsbWw1N3MweQ%3D%3D&filters=WyJbXCJmaWxlX2Zvcm1hdF9jbGFzc1wiLFtcIjFcIl1dIl0%3D&property=ZmlsZV9mb3JtYXRfY2xhc3M%3D)

# Prognostic Model Data

| Variable              | Category         | Total N = 2005 | Training Set n = 1002 | Test Set n = 1003 | MOST N = 1155  |
|-----------------------|-----------------|----------------|-----------------------|-------------------|-----------------|
| **BMI**               | Less than 25    | 644            | 329                   | 315               | 234             |
|                       | 25–29.9         | 814            | 409                   | 405               | 478             |
|                       | 30+             | 547            | 264                   | 283               | 443             |
| **Family History**    | No              | 1601           | 800                   | 801               | 328 [352]       |
|                       | Yes             | 404            | 202                   | 202               | 475             |
| **Ever Injured Knee** | No              | 1239           | 621                   | 618               | 671             |
|                       | Yes             | 766            | 381                   | 385               | 484             |
| **History of Falling**| No              | 1341           | 624                   | 717               | 130 [1003]      |
|                       | Yes             | 664            | 378                   | 286               | 22              |
| **Gender**            | Male            | 895            | 373                   | 522               | 693             |
|                       | Female          | 1110           | 629                   | 481               | 462             |
| **WOMAC**             | —               | 0–82 (6.8)     | 0–71 (8)              | 0–82 (5)          | 0–82 (10) [4 8] |
| **KOA**               | Censored        | 1839           | 913                   | 926               | 1004            |
|                       | Develop KOA     | 166            | 89                    | 77                | 151             |

### Image

- [x-ray](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)

![image](https://storage.googleapis.com/kagglesdsdata/datasets/1257880/2097406/test/0/9003175L.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250918%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250918T113005Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=6c2cec83c1f33eb394f9a8702bd3786e0160a745e921898235589353b469fa239f5a84559c99d3dc9685ba2c117073a0de00114185b5193c6f876f4a8b34ce93dccc46176f8b02a7d5e4acc2a6884a345431368e2d478066192f1dfa7b251de92d74e66beb03f3fa6c3ada7312b6bf59072e55f046f7a5d711a3a8cb8b11cb0f0313f5c345d5cc2e259642744007a2dfc3809cd2d7e8c932d13b505e52a4d20e30dd355914b50682750271b5671f78fe18af2d17728c5a68740d91292a05dcc8582c9456eaa47efc9bd67593402b6da1befa746ec266ddb71134aeed9621e7b4755791b8ec108de4fc7fd2c678fb33ef53a36d4ecb6d78aa0eb0d51e5f66172f)

## Fetus Health

### Numerical

- [patient-information](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/data)

| baseline_value | accelerations | fetal_movement | uterine_contractions | light_decelerations | severe_decelerations | prolongued_decelerations | abnormal_short_term_variability | mean_value_of_short_term_variability | percentage_of_time_with_abnormal_long_term_variability | mean_value_of_long_term_variability | histogram_width | histogram_min | histogram_max | histogram_number_of_peaks | histogram_number_of_zeroes | histogram_mode | histogram_mean | histogram_median | histogram_variance | histogram_tendency | fetal_health |
|----------------|---------------|----------------|----------------------|---------------------|----------------------|--------------------------|---------------------------------|--------------------------------------|-------------------------------------------------------|------------------------------------|----------------|---------------|---------------|---------------------------|---------------------------|----------------|----------------|------------------|-------------------|-------------------|--------------|
| 120.0          | 0.0           | 0.0            | 0.0                  | 0.0                 | 0.0                  | 0.0                      | 73.0                            | 0.5                                  | 43.0                                                  | 2.4                                | 64.0           | 62.0          | 126.0         | 2.0                       | 0.0                       | 120.0          | 137.0          | 121.0            | 73.0               | 1.0               | 2.0          |
| 132.0          | 0.006         | 0.0            | 0.006                | 0.003               | 0.0                  | 0.0                      | 17.0                            | 2.1                                  | 0.0                                                   | 10.4                               | 130.0          | 68.0          | 198.0         | 6.0                       | 1.0                       | 141.0          | 136.0          | 140.0            | 12.0               | 0.0               | 1.0          |
| 133.0          | 0.003         | 0.0            | 0.008                | 0.003               | 0.0                  | 0.0                      | 16.0                            | 2.1                                  | 0.0                                                   | 13.4                               | 130.0          | 68.0          | 198.0         | 5.0                       | 1.0                       | 141.0          | 135.0          | 138.0            | 13.0               | 0.0               | 1.0          |

### Image

- [ultrasound](https://www.kaggle.com/datasets/orvile/ultrasound-fetus-dataset)

![image](https://storage.googleapis.com/kagglesdsdata/datasets/6973444/11339674/Ultrasound%20Fetus%20Dataset/Ultrasound%20Fetus%20Dataset/Data/Data/Datasets/benign/100_HC.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250917%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250917T153433Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=83c4241f44e3803889b29a07fa4222e77d3bb42b8a73204ff292d0bbadb7a195fd20dd05c812c566f1e1839cfffa9dc49f0c7f5e6e4ea273addbe5ca3d549aa4b403401a9c2fdd9e2eb44f8ccd6a52b47d5aae3a1419b07a44dcff2cb43618fec1f020bfce62d84884b6743d798502b079a4b1dd9e267aa6045fe3f8fd9311df98ce2e0648d4b5e6490832307ccd7a2ea8c4cdd261e259b6f1a1ecf23548f2e84ba9cdf0967dd70b2455399fd5eefc6b8c365bd5be8c08f0ec04b81cea08a3a123a5542e763c4d724e0d4e7a85b3d8363cfb3b3b7cea70ed1618bde7fe7eaacc4c64e2ed20194176a406d03b2cd9784b6358ac9d17ce73f2033159cc4e2a489f)

## Thyroid

### Numerical

- [patient-information](https://www.kaggle.com/datasets/emmanuelfwerr/thyroid-disease-data)

| age | sex | on_thyroxine | query_on_thyroxine | on_antithyroid_meds | sick | pregnant | thyroid_surgery | I131_treatment | query_hypothyroid | query_hyperthyroid | lithium | goitre | tumor | hypopituitary | psych | TSH_measured |  TSH | T3_measured |  T3  | TT4_measured | TT4  | T4U_measured | T4U | FTI_measured | FTI | TBG_measured | TBG | referral_source | target | patient_id  |
|-----|-----|--------------|---------------------|----------------------|------|----------|----------------|----------------|-------------------|--------------------|---------|--------|-------|----------------|-------|--------------|------|-------------|------|---------------|------|---------------|-----|---------------|-----|---------------|-----|----------------|--------|-------------|
| 29  | F   | f            | f                   | f                    | f    | f        | f              | f              | t                 | f                  | f       | f      | f     | f              | f     | t            | 0.3  | f           | f    | f             | f    | f             | f   | f             | f   | other          | -      | 840801013   |
| 29  | F   | f            | f                   | f                    | f    | f        | f              | f              | f                 | f                  | f       | f      | f     | f              | f     | t            | 1.6  | t           | 1.9  | t             | 128  | f             | f   | f             | f   | other          | -      | 840801014   |
| 41  | F   | f            | f                   | f                    | f    | f        | f              | f              | f                 | t                  | f       | f      | f     | f              | f     | f            | —    | f           | —    | f             | —    | t             | 11  | —             | —   | other          | -      | 840801042   |

- [patient-information](https://www.kaggle.com/datasets/jainaru/thyroid-disease-data)

| Age | Gender | Smoking | Hx Smoking | Hx Radiotherapy | Thyroid Function | Physical Examination         | Adenopathy | Pathology      | Focality   | Risk |  T   |  N  |  M  | Stage | Response      | Recurred |
|-----|--------|---------|------------|-----------------|------------------|-----------------------------|------------|----------------|------------|------|------|-----|-----|-------|---------------|----------|
| 27  | F      | No      | No         | No              | Euthyroid        | Single nodular goiter-left  | No         | Micropapillary | Uni-Focal  | Low  | T1a  | N0  | M0  | I     | Indeterminate | No       |
| 34  | F      | No      | Yes        | No              | Euthyroid        | Multinodular goiter         | No         | Micropapillary | Uni-Focal  | Low  | T1a  | N0  | M0  | I     | Excellent     | No       |
| 30  | F      | No      | No         | No              | Euthyroid        | Single nodular goiter-right | No         | Micropapillary | Uni-Focal  | Low  | T1a  | N0  | M0  | I     | Excellent     | No       |

### Image

- [ultrasound](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images?select=111.xml)

![image](https://storage.googleapis.com/kagglesdsdata/datasets/1498729/2476474/100_1.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250916%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250916T224958Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=56d52c581e9003a1fed58fab7c3590074e2928e8fa9b6dba502112b30bfc747e6fa373411611ddfb1908991b95a677e5054ade1f91288cc687ea2f9ef437a61494521ff8f5a735341d6fa6b3b6aeaa74f6e0af5dbe07821ab41aaeafa1ba288a460c6c10e06353997565704f29cd79371999cdd7a8571b40f4e774d74698952f0d324f74b869de35328c4eb631c1888b081cfaa042c05082f182911fc320c3f074a8d382cd26edf37392554187982780b37aed7632fbd3da7550412d4201de8a8d225434386f5b43504fb3437f339cb4bf9e3c4e5008d93a1a6d0e9516debae1d2f7e9afcb8c91b73e3f64705f65504c8326d91f3bc8436c5625610eb028594e)

## Kidney Stones

- [patient-information](https://www.kaggle.com/datasets/vuppalaadithyasairam/kidney-stone-prediction-based-on-urine-analysis)

| gravity |  ph  | osmo | cond | urea | calc | target |
|:--------:|:----:|:----:|:----:|:----:|:----:|:------:|
| 1.021    | 4.91 | 725  | 14   | 443  | 2.45 |   0    |
| 1.017    | 5.74 | 577  | 20   | 296  | 4.49 |   0    |
| 1.008    | 7.2  | 321  | 14.9 | 101  | 2.36 |   0    |

- [patient-information](https://www.kaggle.com/datasets/harshghadiya/kidneystone)

| index | gravity |  ph  | osmo | cond | urea | calc | target |
|:-----:|:--------:|:----:|:----:|:----:|:----:|:----:|:------:|
|   0   | 1.021    | 4.91 | 725  | 14.0 | 443  | 2.45 |   0    |
|   1   | 1.017    | 5.74 | 577  | 20.0 | 296  | 4.49 |   0    |
|   2   | 1.008    | 7.20 | 321  | 14.9 | 101  | 2.36 |   0    |

### Image

- [ultrasound](https://www.kaggle.com/datasets/imtkaggleteam/kidney-stone-classification-and-object-detection)

![image](https://storage.googleapis.com/kagglesdsdata/datasets/7794764/12362981/Normal/Normal_1.JPG?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250918%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250918T082431Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=b2bca3698eadaf62a35ba2890fda0ffcda0e848e6fa79cbc379748530193f958c865357c518f671e8b17f02615e091f16f68441a85892f3f00aaeb23238c6f863f4244adee3a4ef305e0e425381dd0d5494f96077e1b9058c8c7938c4349c5d1f220c2f8099cd12a645e92ebca54f8c87da781524057bb33429e58c3d1503c6ab5c4509d4363f28f66e1ee7ffa298709b73c63045d36ff175310ea01bf464e5a35f0730f318e0eecaca2fbf294b6f151d516502897f585b38cba9e6cb86dec02eab636c7995981e6c44bc583f5a5cfa7b10d89336e93ffb33001d4162265a58f8e7720060e5da24ac79947d6f26c459f7dcc25cf83e32008abb06a5ee9e1c5ee)

- [ct-scan](https://www.kaggle.com/datasets/safurahajiheidari/kidney-stone-images)

![image](https://storage.googleapis.com/kagglesdsdata/datasets/3636344/6319067/valid/images/1-3-46-670589-33-1-63703718086220125900001-5014104694799407369_png_jpg.rf.964d590909d71e836492f7dcc4cb9fc5.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250918%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250918T113946Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=4fa666c8062a3e8dc22af20612f54f8d83a85106d861b2dd9640a8615edfb61b8a455d275bec827365d9a43802785adb757926914b8f1799864f968b6c32399ede8961c9cf345b5832a1faed1385d4cfd3bda0b162cefe25b2aa0301371c0f238376689ed5214672c389d116760f435fb7f370c1c806403aef4ddf4f0a711c6b48d5c4e117f0d6dfda602ebb1b7485c041de069b5b297c2205514c02b0515875340b04f7b3589e3ec4f9aec34304a877b4b74f231e39ac018b462e7b8b7b62334c5c33f4424ad6c4d5cec001f1c62194b8bd6e8a60c89309f8a82c19f71fe36a3d57456fed0bbd0e1932ae6af570a225fcd2c7301ae5cfe9149a6b393e031277)

## PCOS

- [patient-information](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos)

> [!NOTE]
> The table below displays only 6 columns for illustration purposes. The full dataset contains a total of 45 columns.

| Sl. No | Patient File No. | PCOS (Y/N) | Age (yrs) | Weight (Kg) | Height (Cm) | BMI       | Blood Group | Pulse rate (bpm) | RR (breaths/min) |
|--------|------------------|------------|-----------|-------------|-------------|-----------|--------------|------------------|------------------|
| 1      | 1                | 0          | 28        | 44.6        | 152         | 19.3      | 15           | 78               | 22               |
| 2      | 2                | 0          | 36        | 65          | 161.5       | 24.921163 | 15           | 74               | 20               |

### Image

- [ultrasound](https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images)

![image](https://storage.googleapis.com/kagglesdsdata/datasets/2065225/3426692/data/train/infected/img1.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250916%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250916T101108Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=20da639dfd52684b7d8333c28673583224850d7f4fb6feea06e3fd4cdca354ce837587dd9b3b276a922205e4dbf8aa4a1268c8b2038c3db05c1e2cae729ba2992ca060a39e928f061cd5894f4dad9dca55fa70f3298c24ab0d12591fac7eb20ae8666bf6b82dfd7013cc0992d1eceea9eda4f4f44dd3ee43d29d32ce35e0634c8a0b9a24b0ebe6e5b07e8a11e25bfdbb16701fe5d84fc92a6f886b31aef5deb1efc4479c8dbb9d1dd06c4de698a07cd2a97f9d1d93bf8b9eff35a11041667484e9a3cef1b1a16b91d657a033aec2c95758e880b40361bab642b2c6b7bdfa102bdb9fa428bcb38bf02115b395b33389f7e6a233b3a28a7feec9400dbe3dc9610c)

## Multiple Sclerosis

### Numerical

[patient-information](https://www.kaggle.com/datasets/desalegngeb/conversion-predictors-of-cis-to-multiple-sclerosis)

### Image

**About the Dataset**  
This dataset contains brain MRI scans (axial, sagittal, and combined views) from 72 MS patients and 59 healthy controls, collected at Ozal University Medical Faculty (2021).  
- **Images**: 1652 axial, 1775 sagittal, 3427 combined (all resized to 224 × 224).  
- **Classes**: Multiple Sclerosis (MS) vs. Healthy.  
- **Features**: Designed for both deep learning (CNN) and feature engineering approaches.  
  **Dataset Link**: [Kaggle – Multiple Sclerosis by BurakTaçci](https://www.kaggle.com/datasets/buraktaci/multiple-sclerosis)

## Skin Lesions

[PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1): a skin lesion dataset composed of patient data and clinical images collected from smartphones

#### Numerical Data

|patient_id|lesion_id|smoke|drink|background_father|background_mother|age|pesticide|gender|skin_cancer_history|cancer_history|has_piped_water|has_sewage_system|fitspatrick|region |diameter_1|diameter_2|diagnostic|itch |grew |hurt |changed|bleed|elevation|img_id               |biopsed|
|----------|---------|-----|-----|-----------------|-----------------|---|---------|------|-------------------|--------------|---------------|-----------------|-----------|-------|----------|----------|----------|-----|-----|-----|-------|-----|---------|---------------------|-------|
|PAT_1516  |1765     |     |     |                 |                 |8  |         |      |                   |              |               |                 |           |ARM    |          |          |NEV       |False|False|False|False  |False|False    |PAT_1516_1765_530.png|False  |
|PAT_46    |881      |False|False|POMERANIA        |POMERANIA        |55 |False    |FEMALE|True               |True          |True           |True             |3.0        |NECK   |6.0       |5.0       |BCC       |True |True |False|True   |True |True     |PAT_46_881_939.png   |True   |
|PAT_1545  |1867     |     |     |                 |                 |77 |         |      |                   |              |               |                 |           |FACE   |          |          |ACK       |True |False|False|False  |False|False    |PAT_1545_1867_547.png|False  |

#### Image Data

<img src="skin-lesion/examples/images/PAT_1516_1765_530.png" alt="PAT_1516" width="200"/>
<img src="skin-lesion/examples/images/PAT_46_881_939.png" alt="PAT_1516" width="200"/>
<img src="skin-lesion/examples/images/PAT_1545_1867_547.png" alt="PAT_1516" width="200"/>

# Final Decision

✅ After reviewing all datasets the chosen one was **Skin Lesions**
