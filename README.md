# User-and-Entity-Behavior-Analytics-UEBA
User and Entity Behavior Analytics by deep learning.  
Detecting users anomalous behaviors from users' daily records. 

内部威胁检测
## Details
All data were extracted from **CERT/R4.2** &nbsp;(*ftp://ftp.sei.cmu.edu/pub/cert-data*)

**Data**: data for detection.  

## Dependent Libraries
- python 3.63-64-bit 
- numpy 1.16.4
- tensorflow 1.8.0
- keras 2.2.2
- sklearn 0.19.1

## Useage
- Run python files step by step.
- Note that **3-Action_Sequence_Training.py** and **4-Static_Feature_Training.py** need to be run for different users separately, you can find the user_sets and change it. **2-Training_Data_Generating.py** also needs to be run under two feature types, you can find the "types" and change it. 

*The provided features and deep learning models in this project are very simple samples, and you can add or create your own features and models based on this project.* : )

## Cite this work
This project is a part of our work that has been published in the ACM/IMS Transactions on Data Science. You can cite this work in your researches. 

ACM/IMS Transactions on Data Science, Volume 1, Issue 3 September 2020, Article No.: 16, pp 1–19 https://doi.org/10.1145/3374749

[Paper Link](https://dl.acm.org/doi/10.1145/3374749)

