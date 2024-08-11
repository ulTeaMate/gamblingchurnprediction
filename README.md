
# Gambling Customer Churn Prediction

Businesses lose and acquire customers constantly. Every business will experience churn, but
the relative size of the churn, also known as the churn rate, is an important indicator of business 
success. Preventing churn is a major performance objective for every business.
Churn rate, also known as attrition rate, is the rate at which customers leave over a given time 
period. Low churn rate means you lose a low proportion of your customers. Low churn implies 
the business is using effective marketing strategies and customers overall are happy. 
Importantly, it is easier and cheaper to keep the customers you already have than to acquire 
new customers. Subsequently, it is pivotal to know why and when your customers tend to 
churn.
Churn rate = (players that left over a period / players at the beginning of period) *100
This project aims to predict customer churn. For business it is extremely valuable to prevent 
churn by intercepting potential churning customers before they churn. Insights in customer 
behaviour can lead to more effective marketing strategies.
## Demo

Gambling Customer Churn Prediction App

deployed at streamlit community

Streamlit (gamblingchurnprediction.streamlit.app)

![Image](https://github.com/ulTeaMate/gamblingchurnprediction/blob/main/app%20prediction%20output%20screenshot.png)




## Author

- [@ulTeamate](https://github.com/ulTeaMate)



## Documentation

[Documentation](https://github.com/ulTeaMate/gamblingchurnprediction)


## Installation

Install my-project with

```bash
  import pandas as pd
import streamlit as st
import joblib
from prediction import predict
from streamlit_echarts import st_echarts
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
```
    
## Project Organization
## Deployment

To deploy this project run

```bash
  joblib
1.4.2
Detected automatically on Aug 11, 2024 (pip) · requirements.txt · BSD-2-Clause AND BSD-3-Clause
matplotlib
3.9.1
Detected automatically on Aug 11, 2024 (pip) · requirements.txt · PSF-2.0
pandas
2.2.2
Detected automatically on Aug 11, 2024 (pip) · requirements.txt · BSD-2-Clause AND BSD-3-Clause
scikit-learn
1.5.1
Detected automatically on Aug 11, 2024 (pip) · requirements.txt · BSD-2-Clause AND BSD-3-Clause
seaborn
0.13.2
Detected automatically on Aug 11, 2024 (pip) · requirements.txt · BSD-2-Clause AND BSD-3-Clause
streamlit-authenticator
0.3.3
Detected automatically on Aug 11, 2024 (pip) · requirements.txt · MIT
streamlit-echarts
0.4.0
Detected automatically on Aug 11, 2024 (pip) · requirements.txt
```


## Appendix

trained machine learning model - SVM rbf tests as best model and therefore used for the prediction app:

```bash
 # Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the SVM model
svm = SVC(kernel='rbf', random_state=42, C=0.1, probability=True)
svm.fit(X_train, y_train)

# Evaluate the model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
```

