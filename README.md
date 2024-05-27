# [Kaggle] Diabetic Retinopathy Classification
> Kaggle에 공개된 diabetic retinopathy 환자들의 fundus image dataset들을 CNN architecture로 classification
<br/>

## Table of Contents
- **[Data](#Data)**
  * **[Data Description](#Data-Description)**
  * **[Data Preprocessing](#Data-Preprocessing)**
- **[CNN Classification](#CNN-Classification)**
  * **[ResNet50](#ResNet50)**
  * **[DenseNet121](#DenseNet121)**
  * **[EfficientNet B3](#EfficientNet-B3)**
<br/><br/>

## Data
### Data Description
* **List of used dataset**
  * EyePACS dataset 2015
  * APTOS 2019
  * INDIAN DIABETIC RETINOPATHY IMAGE DATASET (IDRID)
  * Messidor 2
* **Data citation**
  * EyePACS dataset 2015
    - Emma Dugas, Jared, Jorge, Will Cukierski. (2015). Diabetic Retinopathy Detection. Kaggle. https://kaggle.com/competitions/diabetic-retinopathy-detection
  * APTOS 2019
    - Karthik, Maggie, Sohier Dane. (2019). APTOS 2019 Blindness Detection. Kaggle. https://kaggle.com/competitions/aptos2019-blindness-detection
  * IDRID
    - Prasanna Porwal, Samiksha Pachade, Ravi Kamble, Manesh Kokare, Girish Deshmukh, Vivek Sahasrabuddhe, Fabrice Meriaudeau, April 24, 2018, "Indian Diabetic Retinopathy Image Dataset (IDRiD)", IEEE Dataport, doi: https://dx.doi.org/10.21227/H25W98.
    - https://www.kaggle.com/datasets/gami4388/diabetic-retinopathy-resized-train-15-19-dg?select=resized_train_15_19_DG
  * Messidor 2
    - Abramoff et al, Automated analysis of retinal images for detection of referable diabetic retinopathy, JAMA Ophthalmol. 2013;131:351-7, and in Abramoff et al, Improved automated detection of diabetic retinopathy on a publicly available dataset through integration of deep learning, IOVS. 57:5200-06.
    - https://medicine.uiowa.edu/eye/abramoff
    - https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess/data<br/>
* **Diabetic retinopathy (당뇨병성 망막병증)란**
  * 당뇨병의 합병증으로 발생
  * 초기 증상이 없어 이후 실명 등을 유발할 수 있음
* **본 데이터에는 5가지 category의 fundus image가 존재**<br/> : No diabetic retinopathy(DR) / Mild non-proliferative DR (NPDR) / Moderate NPDR / Severe NPDR / Proliferative DR (PDR) <br/>
<img src=https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/b50c1eed-2fa6-4106-8af8-3e686557c8c7 width="800" height="300"><br/>
  (image source: Asia A-O, Zhu C-Z, Althubiti SA, Al-Alimi D, Xiao Y-L, Ouyang P-B, Al-Qaness MAA. Detection of Diabetic Retinopathy in Retinal Fundus Images Using CNN Classification Models. Electronics. 2022; 11(17):2740. https://doi.org/10.3390/electronics11172740)
* **Dataset Distribution**<br/>
<img src=https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/46c226e2-51bb-4d9f-b7ac-e72800c722ad width="300" height="300"><br/>
  * No DR(0): 68333 (72.22%)
  * Mild NPDR(1): 6870 (7.26%)
  * Moderate NPDR(2): 14667 (15.50%)
  * Severe NPDR(3): 2448 (2.59%)
  * PDR(4): 2306 (2.44%)
* **데이터의 문제점**
  * 심한 데이터 불균형 (No DR이 거의 대부분, Severe NPDR과 PDR이 매우 적음)
  * 흐리거나 초점이 불분명한 사진이 많음
    <p align="left">
     <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/0d4c0c89-4b18-47b6-890e-5814fa11290e" width="20%">
     <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/ed95b2a2-50ff-40c0-8ba7-aa5bc947d01d" width="20%">
     <figcaption align="left"> Eyelid가 보이고 초점이 흐릿한 사진 (왼쪽) / 이미지 정보가 거의 없는 사진 (오른쪽)</figcaption>
    </p>
<br/><br/>

### Data Preprocessing
* **참고**: [https://yhu0409.tistory.com/10](https://yhu0409.tistory.com/10)
1. 매우 흐리거나 초점이 불분명한 사진 삭제
2. Normal data의 20%만 사용 (Downsampling)
3. 전처리 후 남은 data의 distribution
   * No DR: 14922
   * Mild NPDR: 6787
   * Moderate NPDR: 14257
   * Severe NPDR: 2333
   * PDR: 2208
4. Train/Validation/Test data로 나눔
   * No DR, Mild NPDR, Moderate NPDR은 6:2:2로 나누기
   * Severe NPDR, PDR의 경우 3:2:2로 나눈 후, Train set은 추가 data augmentation 진행 (데이터 불균형 막기 위한 upsampling)
5. Train dataset의 Severe NPDR, PDR의 data augmentation 시행
   * 시행 방법<br/>
<img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/3d84ce35-2b17-4b9d-891a-8bc75c4ee138" width="300" height="180"><br/>
   * 시행 횟수: 원래 데이터 수의 2배만큼 시행
6. Downsampling, upsampling 완료 후 Train/Validation/Test dataset의 distribution
<p align="left">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/bcadb749-ff10-4067-bc48-7ba1428e6436" width="200" height="200">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/d1d70edc-28df-48de-8bac-736e0609d30f" width="200" height="200">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/a68bdb7f-af58-4d19-b1d0-0ccde9b29df2" width="200" height="200">
 <figcaption align="left">Train dataset (왼쪽) / Validation dataset (중간) / Test dataset (오른쪽)</figcaption>
</p><br/>
7. 최대한 원형의 이미지 데이터 살리기 위해 image의 대각선 길이만큼 회전 후 padding / 이후 원형으로 crop / 다시 원래대로 회전시키기
<p align="left">
  <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/60e38f34-3672-4297-8099-a8f0ffd25a34" width="540" height="360">
  <figcaption align="left">시행 전 (왼쪽) / 시행 후 (오른쪽)</figcaption>
</p><br/>
8. Rescale
  * Train dataset인 경우 추가 augmentation을 위해 600x600으로 rescale
  * Validation/Test dataset인 경우 512x512로 rescale
9. Ben Graham's preprocessing: Gaussian blur를 취한 후 weight를 줘서 원래 이미지와 subtract -> 조명 효과 제거
<p align="left">
  <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/314b2dfa-1519-4ab2-9e4b-cc5f621cdde0" width="540" height="360">
  <figcaption> 시행 전 (왼쪽) / 시행 후 (오른쪽) </figcaption>
</p><br/>
<br/> <br/> <br/>

## CNN Classification
### ResNet50
#### (1) Loss and Accuracy
* **Train and Valdiation Accuracy**
<p align="left">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/e41f9b45-b10d-43b2-b0c0-c0df14ccfece" width="32%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/df588943-0011-48a5-abd2-46f21d76c895" width="32%">
</p>
<br/>

* **Train and Valdiation Loss**
<p align="left">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/cd94a3a6-11dd-44c2-8da0-26361e2c5b8e" width="32%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/7a585abb-1e18-4ac7-8d8e-2b090697952e" width="32%">
</p>
<br/>

* **Test Dataset Accuracy = 65.10%**
<br/>

#### (2) Sensitivity and Specificity
<img src=https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/23789780-8762-45be-905c-aa26520db28f width="400" height="200"><br/>

#### (3) Confusion Matrix
<img src=https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/ee6226ea-6ca0-48e7-a452-b6e1d1acf49e width="300" height="300"><br/>

#### (4) ROC Curve
##### 1) One vs. Rest multiclass
<img src=https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/af16c3c4-b2ac-40d8-a03a-83600d6a8b35 width="300" height="300"><br/>
##### 2) One vs. One
<p align="left">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/203730a9-012c-4fd4-bf0c-6e5486f119bb" width="18%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/a273d0bc-9812-47e1-8600-127deb155152" width="18%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/9786c7b1-e7b5-4b14-ac0a-5c6e454943e6" width="18%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/5281d959-b2b3-4d7b-8800-761c65840ac7" width="18%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/6ab610fd-96f1-4d1a-b07a-3c0ace0d61cc" width="18%">
</p> <br/>
<p align="left">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/368a42a2-1470-4de5-a1ea-940ce6ede409" width="18%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/845374ac-4bf5-4d11-aee4-89992cfe0c84" width="18%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/e0cf24ea-d08a-4a78-bf8d-052a3748578d" width="18%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/73a8c366-241e-4127-8e4b-30008be3bafd" width="18%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/09eb3cb4-04e6-4d92-a4ec-412d678acf54" width="18%">
</p> <br/>

#### (5) Predictions
<img src=https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/8473f47d-4248-4f38-8fab-64de3cfc6e93 width="300" height="300"><br/>

#### (6) Grad-CAM
##### 1) Grad-CAM by Layers
* Prediction: No DR <br/>
<img src=https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/b309e923-64df-4279-b2c6-ede9a41860be width="600" height="150"> <br/>
<img src=https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/abcccd3e-e9b5-471d-8965-ee31abd0b35f width="600" height="150"> <br/>
* Prediction: Moderate NPDR <br/>
<img src=https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/e86514a6-e568-4154-b0f6-7856a951c19b width="600" height="150"> <br/>

##### 2) Input image / Grad-CAM / Guided-Backpropagation / Guided Grad-CAM
* Prediction: No DR <br/>
<img src=https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/b0892e01-0c10-4c54-8159-7e43caef82d0 width="800" height="200"> <br/>
* Prediction: Drusen <br/>
<img src=https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/ca1d76dd-c04d-4460-9ef1-f5d8e046c91a width="800" height="200"> <br/>
<img src=https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/bbce99bf-a9ed-4881-9cdf-9f1d99bb4535 width="800" height="200"> <br/>
<br/> <br/>


### DenseNet121
#### (1) Loss and Accuracy
* **Train and Valdiation Accuracy**
<p align="left">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/94f9c92e-609a-4bb0-ad12-27b7e04ac5f2" width="32%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/cdd8269c-5f9b-48b7-80f5-46510c2149f9" width="32%">
</p>
<br/>

* **Train and Valdiation Loss**
<p align="left">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/417075a0-b586-46f4-8828-3c8057769845" width="32%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/716d8eb5-2113-4830-bcfa-e8e1bc8526aa" width="32%">
</p>
<br/>

* **Test Dataset Accuracy = 69.16%%**
<br/>

#### (2) Sensitivity and Specificity
<img src=https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/929832d6-8c1b-4dee-b1d6-260ad9c6ff84 width="400" height="200"><br/>

#### (3) Confusion Matrix
<img src=https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/7b18d951-434f-448e-9770-10e02442c13c width="300" height="300"><br/>

#### (4) ROC Curve
##### 1) One vs. Rest multiclass
<img src=https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/0fe18b08-6dd1-4b9a-abc5-9ec6d5360de7 width="300" height="300"><br/>
##### 2) One vs. One
<p align="left">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/48cc978a-7cc4-4d6d-99b2-21cd9e934e3e" width="18%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/0e5acee5-9309-49ed-9788-089a0d7890df" width="18%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/75e4bf8b-7270-4ef5-8d59-6f36bbe551d4" width="18%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/5333fb4b-bf01-4139-8ffd-230520ac600d" width="18%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/df7ae6fd-3115-4069-a24d-753e8e995c46" width="18%">
</p> <br/>
<p align="left">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/921792c1-46ae-4dba-8bfb-429ed71efa57" width="18%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/fdb2e765-860c-4972-ba56-ea899da449a4" width="18%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/b7ed86c2-c0da-4f2d-8d41-b02c188ab936" width="18%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/e0cbb4c6-2fde-46fc-a8cf-dd76ca006b8f" width="18%">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/5e8df3b0-3cf0-482f-850d-d3dfe9073fbb" width="18%">
</p> <br/>

#### (5) Predictions
<img src=https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/67e648ca-a95d-4bee-bdbf-b52863f2619c width="300" height="300"><br/>

#### (6) Grad-CAM
##### 1) Grad-CAM by Layers
* Prediction: No DR <br/>
<img src= width="600" height="150"> <br/>
<img src= width="600" height="150"> <br/>
* Prediction: Moderate NPDR <br/>
<img src= width="600" height="150"> <br/>

##### 2) Input image / Grad-CAM / Guided-Backpropagation / Guided Grad-CAM / Grad-CAM++ / Guided Grad-CAM++
* Prediction: CNV <br/>
<img src= width="1000" height="200"> <br/>
* Prediction: Drusen <br/>
<img src= width="1000" height="200"> <br/>
* Prediction: DME <br/>
<img src= width="1000" height="200"> <br/>
<br/> <br/>
