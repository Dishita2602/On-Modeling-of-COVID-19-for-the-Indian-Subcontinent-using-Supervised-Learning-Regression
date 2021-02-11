# On-Modeling-of-COVID-19-for-the-Indian-Subcontinent-using-Supervised-Learning-Regression
COVID-19, a recently declared pandemic by WHO has taken the world by storm causing catastrophic damage to human life. The novel cornonavirus disease was first incepted in the Wuhan city of China on 31st December 2019. The symptoms include fever, cough, fatigue, shortness of breath or breathing difficulties, and loss of smell and taste. Since the devastating phenomenon is essentially a time-series representation, accurate modeling may benefit in identifying the root cause and accelerate the diagnosis. In the current analysis, COVID-19 modeling is done for the Indian subcontinent based on the data collected for the total cases confirmed, daily recovered, daily deaths, total recovered and total deaths. The data is treated with total confirmed cases as the target variable and rest as feature variables. It is observed that Support vector regressions yields accurate results followed by Polynomial regression. Random forest regression results in overfitting followed by poor Bayesian regression due to highly correlated feature variables. Further, in order to examine the effect of neighboring countries, Pearson correlation matrix is computed to identify geographic cause and effect.

Steps performed :
Step 1 - Data Exploration
Step 2 - Applying machine learning algorithms like Polynomial Regression, Support Vector Regression, Naive Bayes Regression and Random Forest Regression
Step 3 - Comparing Results

We have also try to find correlation of India with its neighboring countries

Conclusion made :
Polynomial regression gave good accuracy but total number of confirmed cases has to go down after acheiving peak which is not possible in these case.
Support Vector Regression has performed extremely well and can be relate with real life.
Naive Bayes Regression has given very poor accuracy.
Random Forest Regression has overfit the data given in ipynb file.
