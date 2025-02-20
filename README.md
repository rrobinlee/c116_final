# STATS C116 Final

## Using Bayesian Analysis and Inference to Identify Medical Conditions and Patient Characteristics that Increase the Mortality Risk of COVID-19 Patients

**Author:** *Robin Lee*, *Ahmed Awadalla*

**Date:** *December 10, 2022*

## Abstract

With the onset of COVID-19 in 2020, doctors and nurses worldwide faced immense challenges in determining which patients required priority treatment. At the peak of the pandemic, limited supplies and resources made it impossible to provide optimal care for every patient. As a result, hospital staff had to make difficult decisions by predicting each patient’s mortality risk.

This study aims to develop a predictive model for mortality risk based on underlying medical conditions. We analyze data from 4,611 patients in the United States who were laboratory-confirmed cases of COVID-19, some of whom survived while others did not. Using Bayesian analysis, we examine the impact of various pre-existing conditions on COVID-19 mortality rates. Because the virus is generally less deadly among young, healthy individuals, we specifically focus on patients with common pre-existing medical conditions to determine which factors contribute most to mortality risk.

To achieve this, we fit a series of logistic regression models to estimate the mortality risk for COVID-19 patients with different underlying conditions. Given the large number of variables, we implement a Horseshoe Prior on the regression parameters to identify the conditions most strongly associated with increased mortality. Finally, we evaluate our model’s accuracy by computing its Expected Log Pointwise Predictive Density (ELPD).

## 1 Introduction

As billions of people across the world prepared to celebrate a brand new year towards the waning days of 2019, the World Health Organization received several incongruous reports from Wuhan, China, regarding an unprecedented illness with pneumonia-like symptoms—the very first glimpses of Coronavirus (COVID-19). In the following months, this newfound virus rapidly spread across the globe, subsequently creating the largest world-wide epidemic in more than a century and affecting millions of lives. Contrary to a majority of other common respiratory viruses, COVID-19 leaves its victims in fluctuating levels of condition, ranging from relatively mild flu-like symptoms to almost certain death. Unfortunately, at the time of this report, more than a million people in the United States have lost their lives due to COVID-related complications. 

Countless doctors and nurses around the world struggled to accurately determine which patients required priority treatment; unfortunately, at the peak of the pandemic, there were simply not enough supplies and resources to provide every victim the utmost care. Thus, hospital staff had to make these difficult decisions by predicting each victim’s mortality risk. In this report, we use Bayesian Statistics to analyze patient data from thousands of victims in the United States in order to determine which factors contribute to COVID-19’s mortality rate the most. By identifying the root cause behind these fatal outcomes, we discern which patients are at a higher risk of severe illness or mortality after contracting COVID, ultimately establishing the appropriate clinical decision-making. 

Because the virus is generally not deadly among young people with no pre-existing illnesses, we specifically examine patients with various common pre-existing medical conditions to predict which ones tend to have the highest COVID-19 mortality risk. Thus, in this report, we examine nearly 5,000 patients—who possess a variety of common medical conditions—around the United States, and utilize Bayesian methods to ascertain the specific characteristics that lead to the highest mortality risk. By fitting a series of logistic regression models, we calculate and predict the mortality risk of COVID-19 patients who also possess other medical conditions. Given the large number of variables, we implement a Horseshoe Prior on the regression parameters to determine which of these conditions contain the highest death rates, before computing our model’s Expected Log Pointwise Predictive Density (ELPD)—allowing us to measure its accuracy. As such, hospitals are able to predict and preemptively distinguish which victims are in significant danger, allowing doctors and nurses to systematically and effectively arrange necessary resources for those who need them the most. 

### 1.1 Mortality Risk Data

The dataset contains information regarding the following COVID-19 patient attributes: “demographics, comorbidities, admission laboratory values, admission medications, admission supplemental oxygen orders, discharge, and mortality”. Because the data is obtained through a healthcare surveillance software package (Streamline Health: *Clinical Looking Glass*), the information in our dataset regarding COVID-19 patients—who have been admitted to a single healthcare system—is a thorough review of their primary medical records. This data is split over a specific period of time, and separated into the first 3 weeks of the pandemic and the following 3 weeks.

Containing about 50 different medical conditions and individual attributes, the dataframe we have selected contains more than 4700 individuals and 85 variables. Several of these predictors include: length of hospital stay (`LOS`), myocardial infraction (`MI`), peripheral vascular disease (`PVD`), congestive heart failure (`CHF`), cardiovascular disease (`CVD`), dementia (`Dement`), Chronic obstructive pulmonary disease (`COPD`), diabetes mellitus simple (`DM simple`), diabetes mellitus complicated (`DM complicated`), oxygen saturation (`OsSats`), mean arterial pressure, in mmHg (`MAP`), D-dimer, in mg/ml (`Ddimer`), platelets, in k per mm3 (`Plts`), international normalized ratio (`INR`), blood urea nitrogen, in mg/dL (`BUN`), alanine aminotransferase, in U/liter (`AST`), while blood cells, in per mm3 (`WBC`) and interleukin-6, in pg/ml (`IL-6`).

Rather than analyze all 84 predictors, we decided to reduce the number of factors to 60. This is because, given the nature of most healthcare institution questionnaires, patients are typically first asked a categorical question—where the response is a discrete “Yes” or “No”, represented in the dataset by `1` and `0` respectively. Thus, we removed most of these binary variables, as they ultimately proved to be quite repetitive. For example, the patients are required to answer whether or not they ever stayed at a hospital (`LOS_Y`), where they either respond “Yes” or “No”; this initial question is succeeded by another variable delineating how many days they stayed in the hospital for (`LOS`), where they respond with a numeric value. Because patients who did not stay at a hospital were simply assigned a `LOS` value of 0, we ultimately did not need a separate categorical variable to help us answer this question—clearly, `LOS =  0` signifies that the person spent 0 days at a hospital and, as such, definitely did not stay. Because many of the discrete-value predictors followed this pattern, we felt compelled to remove these categorical variables in order to tidy our dataset. 

### 1.2 Bayesian Statistical Methods

In order to identify the pre-existing medical conditions with the highest mortality risk, we strive to construct an accurate predictive model by implementing various common model selection strategies—specifically, fitting a sparse model with a “horseshoe” shaped prior on the logistic regression parameters to discern the most efficient fit between the model and the data. The Horseshoe Prior is utilized when there are many potential predictors for a given outcome, but it is not known which ones are actually relevant. By using this technique, we need to first declare a sparsity prior in order to specify the prior estimate of the number of non-zero variables within the dataset—essentially allowing us to predict the number of factors that negatively influence a patient’s mortality risk following their exposure to COVID-19. 

Furthermore, we utilize the expected log pointwise predictive density (ELPD); as such, we are able to calculate the probability of producing a data set from our data generating process. The Expected Log Pointwise Predictive Density (ELPD) is a measure of the quality of a statistical model and how much it accurately fits not only our dataset, but also new data. In other words, if our model has a high probability of producing a dataset similar to our original COVID-19 mortality data and therefore possess a high ELPD value, we are confident it is accurate. Through these methods, we are able to build a predictive model of COVID-19 mortality risk. 

## 2 Analysis and Results

### 2.1 Horseshoe Prior on Regression Parameters

Because we have a large number of predictors in our dataset, we use a Horseshoe Prior approach on our regression parameters—where $(y | \beta) \sim N(\beta,\sigma^2 I)$, and $\beta$ is believed to be sparse—to elucidate the unknown sparsity and handle the significant amount of strong signals. The most notable characteristic of the sparsity prior is that we need to specify the prior estimate of the number of non-zero variables out of the selected 44 predictors; in other words, we want to predict the number of factors that contribute to COVID-related mortality before analyzing the data.

We decide to use this method, because the Horseshoe Prior is particularly useful as a shrinkage prior for sparse problems, allowing us to handle the dataset's relatively unknown sparsity and the number of large outlying signals. Because of its flat, Cauchy-like tails and the distribution’s spike at the origin, the Prior allows the strong signals—variables that noticeably increase a patient’s mortality risk following their exposure to COVID-19—to remain large, while the zero elements of $\beta$ are severely shrunk. As mentioned earlier, the Horseshoe Prior is utilized when there are many potential predictors for the response (`Death`), but we are not sure which ones are actually relevant. The Prior helps us shrink the estimates of the irrelevant predictors towards zero, while preserving the relevant predictors. By diminishing the factors that are not useful, this method allows us to see which variables are sufficient in predicting COVID-19 mortality risk and which are not. 

*Applying the Horseshoe Prior*

In order to implement the Horseshoe Prior, we standardize all the columns and create a new dataset (`covid1`) featuring all the modified data; this ensures that each value in the dataset is on the same numeric scale. Because we are utilizing a Logistic Regression, we do not need to standardize the response variable (`Death`), since all the values are either `0` (Did not die) or `1` (Died). 

```{r, echo = T, warning = F, fig.align='center', fig.width=12, fig.height=10, comment = NA, eval = T}
covid$X <- data.matrix(covid[,2:ncol(covid)])
yf<-covid$Death
Xf<-covid$X
Xf<-t( (t(Xf)-apply(Xf,2,mean))/apply(Xf,2,sd))
n<-length(yf)
i.te<-sample(1:n,100)
i.tr<-(1:n)[-i.te]
y<-yf[i.tr] ; y.te<-yf[i.te]
X<-Xf[i.tr,]; X.te<-Xf[i.te,]
p=dim(X)[2]
covid1 <- as.data.frame(cbind(y,X))
```

With our new dataset, we specify the number of non-zero coefficients (`p_nonzero`). According to the CDC, individuals with obesity, diabetes, chronic lung disease, or sickle cell disease face an increased mortality risk following COVID-19; furthermore, older patients (typically above 65) and those who are immunocompromised are at significant danger as well. Because our dataset contains a large number of predictors, ranging from race to various medical conditions, we believe that there will be about 10 non-zero coefficients—meaning that we estimate that there will be 10 relevant predictors that increase risk. 

Coming to this conclusion, we construct our prior distribution for the regression coefficients using the Hierarchical Shrinkage Family (exemplified by the `hs` function). We utilize this distribution, because it implements a half-Cauchy distributed standard deviation with a median of zero and a half-Cauchy scale parameter (these are the descriptors of the distribution); in other words, this function is the same as the Horseshoe Prior we are trying to apply. 

After specifying our prior guess for the number of relevant variables that we believe will influence the COVID-19 fatality risk, we create our Bayesian Generalized Linear Model using the `stan_glm` function, allowing us to perform a full Bayesian estimation using Markov Chain Monte Carlo (MCMC). Because we have specified the value above, we are able to add priors onto the coefficients using `prior = hs_prior`, where `hs_prior` is the prior distribution for the regression coefficients. Because we want to find the number of non-zero variables out of the selected 44 predictors, we need to fit all 44 variables into the model—starting with a patient’s length of stay at the hospital (`LOS`) and ending with their Troponin levels (`Troponin`). 

It is important to note that even though there are many predictors that increase a patient's risk, we will not implement all non-zero coefficients in our final model. For example, while we are confident that the severity of a patient’s symptoms (‘severity’) and their length of stay at the hospital (‘LOS’) will definitely contribute to an individual’s mortality risk, we will not use these variables in our final model; this is because we are focusing specifically on pre-existing conditions and patient attributes. However, in order to fully analyze the dataset, our initial fit, named `fit`, contains all the parameters in our tidied dataset. 

Applying these aforementioned concepts, we construct the Hierarchical Shrinkage Family and the Horseshoe Prior model. With our Generalized Bayesian model, we are able to plot each variable’s signal and identify which predictors have a non-zero coefficient. As such, we locate the factors that prove to increase a patient’s mortality risk after contracting COVID-19 by distinguishing which predictors deviate from the line at 0. In order to present this data in a detailed and comprehensive manner, we create two plots that essentially both display the coefficients, with the first graph exemplifying each variable’s distribution and the second graph highlighting the actual value of the coefficient. 

Figure 1: Horseshoe Prior on Regression Parameters in COVID-19 Patient Data
![image](https://github.com/user-attachments/assets/ffba2231-e5a0-4c54-a3ae-4ecd74630c33)

From the Figure 1, we can see that the following parameters significantly increase a patient’s mortality risk following their exposure to COVID-19 and are all strong predictors: Peripheral Vascular Disease (`PVD`), End-Stage Renal Disease (`Renal`), Stroke (`Stroke`), Syncope (`OldSyncope`), Age (`Age`), Temperature (`Temp`), Mean Arterial Pressure in mmHg (`MAP`), Alanine Aminotransferase in U/liter (`AST`), Lymphocyte (`Lympho`), Interleukin-6 in pg/ml (`IL-6`), Ferritin (specifically Ferritin > 300: `Ferritin_gt_300`), Procalcitonin (`Procalcitonin`), 	C-reactive Protein (`CrctProtein`), Troponin (`Troponin`). By implementing the Horseshoe Prior, we identify the parameters with the largest signals in relation to the `Death` parameter. Because we are evaluating the mortality risk of COVID-19, the response variable is whether or not a patient has died; thus, we use these variables in a Logistic Regression model to highlight which factors have the highest mortality rate. 

### 2.2 Constructing the Logistic Regression Model

Logistic regression is a type of regression analysis that is typically utilized to predict the outcome of a categorical dependent variable, based on one or more independent variables. In this type of model, the dependent variable is binary, meaning that it can only take on two values (in our model, `Death` is denoted by `0` or `1`). The independent variables we have chosen are either continuous (such as `Stroke`) or categorical (such as `CrctProtein`). These parameters predict the probability that the dependent variable will take on a certain value: `1` or `0`.

Various studies have reported the clinical features of critical patients with COVID-19. In this study, we intend to analyze different clinical features and risk factors to identify the fatal consequences of the disease. Despite scientists' efforts to better understand the clinical features of the disease, the current understanding of the risk factors for COVID-19 is still ongoing, this study does not include every clinical feature of critical patients that have COVID-19; and therefore the study does have limitations. Hamidreza Kouhpayeh—a distinguished member of the Infectious Disease Department at the Zahedan Uiversity of Medical Sciences—found in 2022 that the overall mortality rate from COVID-19 is between 3.77% and 5.4%; however, we see this rate increase to between 41.1% and 61.5% among severe or critical patients $^{[2]}$. In our analysis, we identify risk factors associated with disease severity and mortality in patients with COVID-19.

As mentioned earlier, logistic regression is a standard way to model binary outcomes (that is, response values $y_i$ take the values: `0` or `1`). In the previous section, we have illustrated a logistic regression model using Bayesian ideas, specifically with prior knowledge regarding which variables are strong indicators of mortality; this method is the Horseshoe Prior. For each case $i$, we label $y_i = 1$ if the patient died or $y_i = 0$ if the patient survived. The logistic regression will help us in predicting the likelihood of mortality from COVID-19, given a patient possesses one or more of the variables at question. 

A logistic regression model is fitted to the our dataset with dependent variable—specifically whether a patient survived or not—and the aforementioned independent variables that we selected using the Horseshoe Prior—specifically Peripheral Vascular Disease (`PVD`), End-Stage Renal Disease (`Renal`), Stroke (`Stroke`), Syncope (`OldSyncope`), Age (`Age`), Temperature (`Temp`), Mean Arterial Pressure in mmHg (`MAP`), Alanine Aminotransferase in U/liter (`AST`), Lymphocyte (`Lympho`), Interleukin-6 in pg/ml (`IL-6`), Ferritin (specifically Ferritin > 300: `Ferritin_gt_300`), Procalcitonin (`Procalcitonin`), C-reactive Protein (`CrctProtein`), Troponin (`Troponin`). 

We model the probability that y = 1:

$$Pr(y_i =1) \text{logit}^{-1} (X_iB) ; \text{logit}^{-1}(x) = \frac{e^x}{1+e^x} $$ 

Notice that:
$\text{logit}^{-1}(x)$ transforms continuous values to the range (0,1), which is necessary, since probabilities must be bewteen 0 and 1.

```{r, echo = T, warning = F, fig.align='center', fig.width=8, fig.height=8, comment = NA}
fit_bayes <- stan_glm(y ~ Age + PVD + Renal + Stroke + OldSyncope + Temp + MAP + AST + 
                        Lympho + Ferritin_gt_300 + Procalcitonin + CrctProtein + Troponin, 
                      prior = normal(), prior_intercept = normal(), 
                      family=binomial(link="logit"),data= covid1)
```

We choose `Age`, `PVD`, `Renal`, `Stroke`, `OldSyncope`, `Temp`, `MAP`, `AST`, `Lympho`, `Ferritin_gt_300`, `Procalcitonin`, `CrctProtein`, and `Troponin` as the covariates we want to focus on. We choose these covariates, because our Horseshoe Prior shows that these parameters are strong predictors that significantly increase mortality risk. Furthermore, they are all relatively independent of each other, which means that they contain a lot of non-intersecting predictive capacity.

From our results (not shown), the parameter estimates no longer have test statistics and p-values as in the Frequentist approach. This is because—unlike a Frequentist approach—Bayesian estimation samples from the posterior distribution, which means that instead of a point estimate and a test statistic, we get a distribution of plausible values for the parameters; the estimates section summarizes those distributions. Specifically, we get the mean, standard deviation, and commonly used percentiles. 

The parameters in the estimates section (other than the coefficients we have entered into the model) are sigma—which  represents the standard deviation of errors—amd mean_ppd—the mean of the posterior predictive distribution of our outcome variable, `Death`. Finally, log-posterior is analogous to the likelihood; this represents the log of the combined posterior distributions, which will be used for model comparisons. 

Figure 2: Scrunched Pairs Plot of Logistic Regression for Risk Factors
![image](https://github.com/user-attachments/assets/69862563-7793-43b1-a571-1164ec90ce36)

The above density plots confirm that our estimates are normal, which means that we know the central limit theorem is effective. Our scrunched pairs plot takes a closer look at the distribution of each parameter. Noticeably, each predictor has a peak that is above or below 0, indicating that they are significant. For example, we can see that `Age` and `MAP` are strong indicators for COVID-related mortality.

Figure 3: Factor-Analysis-Based Logistic Regression for risk factors associated with COVID-19 mortality
![image](https://github.com/user-attachments/assets/309f040c-0157-4e7f-bee9-3158dc01d063)

Unlike in the Frequentist regression—where there is always a solution using ordinary least squares, in Bayesian models, we have to check to make sure the model converged. If a model converges, then we are confident that the parameter estimates are stable. Otherwise, our results are unreliable. In Bayesian estimation, posterior distributions are sampled in groups, known as chains. We can measure the stability of our estimates by comparing the variance within chains to the variance across chains, which is denoted by the R-hat statistic. In general, we want all R-hat values to be close to 1 in order to conclude the model has converged, as in this example. Taking a look at Figure 3 (right), we can see our model fits the observed data pretty well, which allows us to make inferences about the data and shows that our model is accurate. 

Further corroborating the belief that our model fits the data well, we calculate the probability of direction—a measure of the likelihood that a given event will occur in a specific direction. To calculate the probability, we identify the so-called event we are interested in (the predictors) and the specific direction we want to assess the likelihood of (the data). Once we identify these two values, we count the number of times the event occurred in that direction. In order to find the percentage, we divide each value associated with a parameter by the total number of times the event occurred, giving the probability of direction between 0 and 1. 

Our analysis suggests that Peripheral Vascular Disease (`PVD`), End-Stage Renal Disease (`Renal`), Stroke (`Stroke`), Age (`Age`), Temperature (`Temp`), Mean Arterial Pressure in mmHg (`MAP`), Alanine Aminotransferase in U/liter (`AST`), Ferritin (specifically Ferritin > 300: `Ferritin_gt_300`), Procalcitonin (`Procalcitonin`), C-reactive Protein (`CrctProtein`), Troponin (`Troponin`) are important risk factors for this disease. These results have important clinical implications, such as clinical management and specific preventive measures for patients with these underlying medical conditions.

In conclusion, the selection of appropriate clinical indicators for early identification and class treatment of COVID-19 patients is very important and can help save lives.


### 2.3 Expected Log Pointwise Predictive Density (ELPD)

After selecting the parameters that appear to increase a patient’s mortality risk, we want to calculate just how accurate our logistic regression model is and compute its prediction accuracy. The Expected Log Pointwise Predictive Density (ELPD) is a measure of the quality of our statistical model and how much it fits not only the current dataset, but also any new data.

By juxtaposing the fit of our logit model to a probit regression—which determines categorical likelihood rather than odds of success (logit)—using the same data, we employ the theoretical Expected Log Pointwise Predictive Density (ELPD) of a new dataset. As such, we utilize cross-validation to compare how well our model prognosticates any new observations and calculate the predictive density values for each observation, ensuring the model’s accuracy.  

In order to understand how this process works, we let $y_1^{new}, ... y_n^{new}$ be a new dataset created by the ELPD process, where the covariates are the same as the original dataset: 

$P(y_i^{new} | y_1,...,y_n) = \int_{}^{} P(y_i^{new} | \theta) P(\theta | y_1, ..., y_n) d\theta, \textrm{ where } i = 1, ..., n$

For each prediction, we are calculating the log-likelihood (probability) of each observation given our model. We obtain the ELPD by averaging the log-likelihood of each observations in the dataset, since it shows how well the model predicts the data. Thus, because we want to construct a model that has a high probability of producing a dataset relatively similar to our original data, we focus on the new dataset’s ELPD after the generating process:

![image](https://github.com/user-attachments/assets/eea9bda1-7428-44e7-8e88-0890d2b1027c)

If our model has a high probability of generating a dataset similar to the one we are using, then we are confident that it should be a good model. In general, a model with a higher ELPD is considered to be a better fit for the data and more likely to make accurate predictions of new data.

We want to use cross-validation (as mentioned above), because the exact computation of the ELPD is mathematically challenging and estimating it is the most efficient method. This procedure essentially removes a single observation from the data and refits our model with the other patients' information. Thus, by using cross-validation, we predict the point we removed in order to show how accurate the model is. 

The natural estimator is: $$\widehat{\text{elpd}} = \Sigma^n_{i=1} \text{log} P(y_i | y_{-i})$$ 

where the posterior probability of the observed $y_i$ is: 

$$P(y_i | y_{-1}) = \int_{}^{} P(y_i | \theta) P(\theta | y_{-i}) d\theta$$.

Because we have more than 4700 independent observations, we decide to only sample 200 patients. This optimizes our ELPD, since the function requires us to cross-validate each observation and predictor. 

```{r, echo = T, warning = F, fig.align='center', fig.width=12, fig.height=10, comment = NA}
covid2 <- sample_n(covid1, 200)
fit1 <- stan_glm(y ~ PVD + Renal + Stroke + OldSyncope + OtherBrnLsn + Age + OSat_lt_94 + 
                   Temp + MAP + MAP_lt_70 + AST + Lympho_lt_1 + IL6_gt_150 + 
                   Ferritin_gt_300 + Procalcitonin + CrctProtein + 
                   Procalciton_gt_0 + Troponin, 
                data = covid2, family=binomial(),
                prior = hs_prior, prior_intercept = t_prior, 
                seed = 1, adapt_delta = 0.99, refresh=0)
```

Using the predictors we found through the Horseshoe prior, the model we are testing looks like such:

*Note*: the values are different from the logistic regression model in section 2.2, because we are applying a Bayesian model like in section 2.1. This means that we are applying a Horseshoe Prior using `stan_glm`.  

In order to measure how well our chosen relevant variables predict the response variable, we plot the ELPD and the root mean squared error (RMSE) of each predictor: 

Figure 4: Expected Log Pointwise Predictive Density Validation
![image](https://github.com/user-attachments/assets/3eabe223-1871-45e1-9006-c45ea942f6b3)

Using the `cv_varsel` method, we are able to implement cross-validation to see how many of variables should be included in our final model. Because we want to measure how accurate our model is and compare $y_i$ (the removed observation) to the posterior predictive for $y_i$, we use the root mean squared error to measure the distance between the actual value and the predicted value. In the plot above (Figure 4), the line measures the overall quality of the fit.

Thus, we need to find the x value that ensures the line is the most optimal; this means that we have to find an ideal number of predictors—which is labeled on the x-axis—that is not only closest to the dotted line, but also does not contain too many variables. We do not want too many variables, because even though the line is extremely close to the dotted line, the quality of the fit actually decreases and is not necessarily more accurate. 

Comparing the Expected Log Pointwise Predictive Density relative to the full model:

Figure 5: Expected Log Pointwise Predictive Density Validation Relative to the Full Model
![image](https://github.com/user-attachments/assets/96841a2f-d269-461f-96ba-62eaafc615ab)

Therefore, after plotting the ELPD and the root mean squared error (RMSE) in Figure 4 and 5, we see the model start to over-fit after about 9 predictors. This is because the number of predictors that is closest to the dotted line is around 9, and we do not want to risk decreasing the quality of our fit by including too many parameters. However, while we believe the ideal number of variables is 9, we decide to use the 11 predictors as mentioned in the Section 2.2 Constructing the Logistic Regression Model; this is because we believe that several of the predictors (such as `Age` and `Stroke`) work hand-in-hand to increase COVID-19 mortality risk. Thus, in order to better encapsulate the data, we implement a few more parameters to our model than recommended. Comparing to the full model (Figure 5), the two plots are extremely similar. 

## 3 Discussion

*Conclusion*

COVID-19 patient’s mortality risk could be predicted by developing a Logistic Regression model with Peripheral Vascular Disease (`PVD`), End-Stage Renal Disease (`Renal`), Stroke (`Stroke`), Syncope (`OldSyncope`), Age (`Age`), Temperature (`Temp`), Mean Arterial Pressure in mmHg (`MAP`), Alanine Aminotransferase in U/liter (`AST`), Lymphocyte (`Lympho`), Interleukin-6 in pg/ml (`IL-6`), Ferritin (specifically Ferritin > 300: `Ferritin_gt_300`), Procalcitonin (`Procalcitonin`),C-reactive Protein (`CrctProtein`), Troponin (`Troponin`) as predictors. 

Further analysis showed that these 11 predictors in particular model risk the best: Peripheral Vascular Disease (`PVD`), End-Stage Renal Disease (`Renal`), Stroke (`Stroke`), Age (`Age`), Temperature (`Temp`), Mean Arterial Pressure in mmHg (`MAP`), Alanine Aminotransferase in U/liter (`AST`), Ferritin (specifically Ferritin > 300: `Ferritin_gt_300`), Procalcitonin (`Procalcitonin`), C-reactive Protein (`CrctProtein`), Troponin (`Troponin`)

The model developed has shown a good performance based on all metrics. It can help hospitals prioritize patients who are really in need and reduce the mortality rate. However, based on other studies our data does not include all clinical features of critical patients with COVID-19,  which may result in a lack of recognizing the pattern. In future studies, gathering more data on different clinical features for training is expected. 

*Limitations and Further Research*

A significant issue that we faced while creating our model involved the Expected Log Pointwise Predictive Density (ELPD). Because we utilized cross-validation to compare our model with any new values, we had to compare each observation and variable, as well as calculate the log-likelihood (probability) of each value given our model. Thus, with 4711 observations and more than 40 parameters, computing the entire dataset creates issues with running the `cv_varsel` function. As such, we had to sample the dataset and utilize only 200 observations. This means that the model might not be the most accurate, since it does not represent the entire data. While the overall data for each variable is normal, this is not necessarily the case for a sample, because we are randomly sampling. 

Furthermore, there are many factors that contribute to COVID’s mortality risk. While we focused specifically on certain medical conditions, there are countless other pre-existing diseases and conditions—such as gender—that work alongside the variables in our dataset to increase fatality. Because there are other significant factors that we failed to include, we are not completely confident that all the predictors we selected in our model actually increase mortality risk, but if there are underlying factors that played a role. 

While we do not have missing data, the data only includes patients from a specific healthcare surveillance software package in the state of Georgia. This means that the individuals we have analyzed are relatively financially and economically stable, because they are able to afford healthcare. Thus, our dataset is not only unrepresentative of the United States, but the world as a whole. Even though we have more than 4711 individuals, this is small in comparison to the total number of people who contracted COVID-19. 

For further research, we want to run an ELPD for the entire dataset with better resources, since our computers were unable to handle beyond 4 cores in the `cv_varsel` function. Taking a look at the code above, we only utilized 2 cores. We also believe that including more factors—such as gender—would improve our model’s accuracy, as well as more data from across the United States and the world. Ultimately, we want our model to be representative of all individuals who contract COVID-19. 


## References

[1] Chowdhury, M. E. H., Rahman, T., Khandakar, A., Al-Madeed, S., Zughaier, S. M., Doi, S. A. R., Hassen, H., & Islam, M. T. (2021, April 21). An early warning tool for predicting mortality risk of COVID- 19 patients using machine learning - cognitive computation. SpringerLink. Retrieved December 9, 2022, from https://link.springer.com/article/10.1007/s12559-020-09812-7

[2] Kouhpayeh, H. (2022, April 12). Clinical Features Predicting COVID-19 Mortality Risk. European journal of translational myology. Retrieved December 9, 2022, from https://www.ncbi.nlm.nih.gov/pmc/ articles/PMC9295175/

[3] The Novel Coronavirus Pneumonia Emergency Response Epidemiology Team. (2020, February 1). The Epidemiological Characteristics of an Outbreak of 2019 Novel Coronavirus Diseases (COVID-19) - China, 2020. China CDC Weekly. Retrieved December 9, 2022, from https://weekly.chinacdc.cn/en/article/id/ e53946e2-c6c4-41e9-9a9b-fea8db1a8f51

[4] WHO China Joint Mission on COVID-19 Final Report. (n.d.). Retrieved December 10, 2022, from https://www.who.int/docs/default-source/coronaviruse/who-china-joint-mission-on-covid-19-final-report

[5] Centers for Disease Control and Prevention. (n.d.). CDC Covid Data Tracker. Centers for Dis- ease Control and Prevention. Retrieved December 9, 2022, from https://covid.cdc.gov/covid-data-tracker/ #demographicsovertime

[6] Age, Sex, Existing Conditions of COVID-19 Cases and Deaths. Worldometer. (n.d.). Retrieved December 9, 2022, from https://www.worldometers.info/coronavirus/coronavirus-age-sex-demographics/

[7] Gelman, A., & Hill, J. (2018). Data Analysis using Regression and Multilevel/Hierarchical Models. Cambridge Univ. Press.
