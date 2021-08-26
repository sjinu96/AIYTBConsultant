# 🤡AI YouTube Consultant

![image](https://user-images.githubusercontent.com/71121461/130922216-28fe4eb8-354b-416d-a963-f1a0325bd967.png)


* 본 저장소는 서울시립대학교 자연과학대학 수학과에서 진행한 캡스톤디자인 프로젝트를 정리하기 위한 공간입니다.
* 유튜브 채널 정보를 바탕으로 구독자 추이를 예측하고, 구독자 수를 늘리기 위한 방안을 제공하는 것이 프로젝트의 목표입니다.

## 주요 진행 상황
### 데이터 구조  
![](https://github.com/iloveslowfood/AIYTBConsultant/blob/master/etc/Presentation/Data%20Structure.png)  

### 모델
* 모델 1 : LSTM, CNN

![image](https://user-images.githubusercontent.com/71121461/130922468-ea0391d7-b512-4b51-bf42-1df54eb66551.png) 
* 모델 2 : Multiple LSTM(w.r.t. Vanilla LSTM, Stock Indicators(SI), and Full model)

![image](https://user-images.githubusercontent.com/71121461/130922565-5bde94fd-b1ac-4489-9831-d1d14085b842.png)
* 모델 3 : Time2Vec(w.r.t CNN and LSTM)

![image](https://user-images.githubusercontent.com/71121461/130923054-acdcdf47-ccfa-4b2a-a01c-3512210cdeba.png)
* 모델 4 : Wavenet

![image](https://user-images.githubusercontent.com/71121461/130923095-44d43420-fb92-4345-ad9c-88d35f6a9c3c.png)
* 모델 5 : ConvGRU

![image](https://user-images.githubusercontent.com/71121461/130923134-63224b2a-b4d8-4023-aeac-f47128b5229e.png)

### 평가(일부)

 ![image](https://user-images.githubusercontent.com/71121461/130924128-69fb9cf5-6be6-4fec-8b52-2c545d4baab9.png)


### 결과(일부)

* SI-MLSTM(for predicting 1 day)
![image](https://user-images.githubusercontent.com/71121461/130923972-e472bf9f-f4d0-43e7-a5d9-e2ccb6decfe9.png)

* Wavenet (for predicting 30 days)
![image](https://user-images.githubusercontent.com/71121461/130923410-967d38a7-f8ec-44e8-aa35-435ba956e53b.png)

* convGRU (for predicting 1 day)
![image](https://user-images.githubusercontent.com/71121461/130924269-7851e75b-fddb-42d4-9d9f-3bdd107d7de6.png)


## 🏃‍♂️프로젝트 기간
* 2020.09~2020.12

## 👀디렉토리 구조
```
AIYTBConsultant
├─data                # 데이터가 저장된 폴더
│  ├─raw              # 수집한 raw 데이터가 저장된 폴더
|  └─train_raw.csv    # 학습에 활용되는 데이터
├─etc                 # 노트북 커널, 각종 분석 플롯 등이 저장된 폴더
├─model               # 모델 모듈 저장 폴더
└─preprocessing       # 전처리 모듈 저장 폴더
```
## 👩‍👧‍👦프로젝트 인원
* 프로젝트 참여자: 고지형(본인), 김양기, 박진수, 안나민
* 담당 교수: 김정래
---
* Capstone projects in the department of mathematics, University of Seoul
* Period: September 2020 ~ December 2020
* Advisor: Prof. Jungrae Kim
* Participants: Jihyeong Ko, Yanggi Kim, Jinsu Park, Namin Ahn

