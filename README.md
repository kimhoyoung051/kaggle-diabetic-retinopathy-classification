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
    - https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess/data
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
   * 시행 방법
<img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/3d84ce35-2b17-4b9d-891a-8bc75c4ee138" width="300" height="400"><br/>
   * 시행 횟수: 데이터의 2배 시행
6. Downsampling, upsampling 완료 후 Train/Validation/Test dataset의 distribution
<p align="center">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/bcadb749-ff10-4067-bc48-7ba1428e6436" width="200" height="200">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/d1d70edc-28df-48de-8bac-736e0609d30f" width="200" height="200">
 <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/a68bdb7f-af58-4d19-b1d0-0ccde9b29df2" width="200" height="200">
</p>
7. 최대한 원형의 이미지 데이터 살리기 위해 image의 대각선 길이만큼 회전 후 padding / 이후 원형으로 crop / 다시 원래대로 회전시키기
<p align="center">
  <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/60e38f34-3672-4297-8099-a8f0ffd25a34" width="600" height="400">
  <figcaption align="center">시행 전 (왼쪽) 시행 후 (오른쪽)</figcaption>
</p><br/>
8. Rescale
  * Train dataset인 경우 추가 augmentation을 위해 600x600으로 rescale
  * Validation/Test dataset인 경우 512x512로 rescale
9. Ben Graham's preprocessing: Gaussian blur를 취한 후 weight를 줘서 원래 이미지와 subtract -> 조명 효과 제거
<p align="center">
  <img src="https://github.com/kimhoyoung051/kaggle-diabetic-retinopathy-classification/assets/164658426/314b2dfa-1519-4ab2-9e4b-cc5f621cdde0" width="600" height="400">
  <figcaption align="center">시행 전 (왼쪽) 시행 후 (오른쪽)</figcaption>
</p><br/>
<br/> <br/> <br/>

## CNN Classification
### ResNet50
#### (1) Loss and Accuracy
* **Train and Valdiation Accuracy**
<p align="left">
 <img src="" width="32%">
 <img src="" width="32%">
</p>
<br/>

* **Train and Valdiation Loss**
<p align="left">
 <img src="" width="32%">
 <img src="" width="32%">
</p>
<br/>

#### (2) Sensitivity and Specificity
<img src= width="400" height="200"><br/>

#### (3) Confusion Matrix
<img src= width="300" height="300"><br/>

#### (4) ROC Curve
##### 1) One vs. Rest multiclass
<img src= width="300" height="300"><br/>
##### 2) One vs. One
<p align="left">
 <img src="" width="15%">
 <img src="" width="15%">
 <img src="" width="15%">
 <img src="" width="15%">
 <img src="" width="15%">
 <img src="" width="15%">
</p>
<br/>

#### (5) Predictions
<img src= width="300" height="300"><br/>

#### (6) Grad-CAM
##### 1) Grad-CAM by Layers
* Prediction: Normal <br/>
<img src= width="600" height="150"> <br/>
<img src= width="600" height="150"> <br/>
* Prediction: CNV <br/>
<img src= width="600" height="150"> <br/>
<img src= width="600" height="150"> <br/>
* Prediction: Drusen <br/>
<img src= width="600" height="150"> <br/>
* Prediction: DME <br/>
<img src= width="600" height="150"> <br/>
##### 2) Input image / Grad-CAM / Guided-Backpropagation / Guided Grad-CAM / Grad-CAM++ / Guided Grad-CAM++
* Prediction: CNV <br/>
<img src= width="800" height="200"> <br/>
* Prediction: Drusen <br/>
<img src= width="800" height="200"> <br/>
* Prediction: DME <br/>
<img src= width="800" height="200"> <br/>

