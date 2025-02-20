require(bayesplot)
require(rstanarm)
require(readxl)
require(dplyr)
require(ggplot2)
require(gridExtra)
library(projpred)
library(bayestestR)

covid <- read_excel("covid_mortality.xlsx")

# Tidying the data

# Removing unnecessary variables
covid <- select(covid, -c(1,2))
covid <- covid %>% select(-contains("Yes")) %>% select(-contains("Score"))

# Rename variables 
colnames(covid)[colnames(covid) == "Age...5"] ="Age_Range"
colnames(covid)[colnames(covid) == "Age...27"] ="Age"
colnames(covid)[colnames(covid) == "DM Complicated"] ="DM_C"
colnames(covid)[colnames(covid) == "DM Simple"] ="DM_S"
colnames(covid)[colnames(covid) == "Renal Disease"] ="Renal"
colnames(covid)[colnames(covid) == "All CNS"] ="All_CNS"
colnames(covid)[colnames(covid) == "Pure CNS"] ="Pure_CNS"
colnames(covid)[colnames(covid) == "O2 Sat < 94"] ="OSat_lt_94"
colnames(covid)[colnames(covid) == "Temp > 38"] ="Temp_gt_38"
colnames(covid)[colnames(covid) == "MAP < 70"] ="MAP_lt_70"
colnames(covid)[colnames(covid) == "D-Dimer > 3"] ="Ddimer_gt_3"
colnames(covid)[colnames(covid) == "INR > 1.2"] ="INR_gt_1.2"
colnames(covid)[colnames(covid) == "BUN > 30"] ="BUN_gt_30"
colnames(covid)[colnames(covid) == "Sodium < 139 or > 154"] ="Sodium_bt_139_154"
colnames(covid)[colnames(covid) == "Glucose <60 or > 500"] ="Glucose_bt_60_500"
colnames(covid)[colnames(covid) == "AST > 40"] ="AST_gt_40"
colnames(covid)[colnames(covid) == "ALT > 40"] ="ALT_gt_40"
colnames(covid)[colnames(covid) == "WBC <1.8 or > 4.8"] ="WBC_bt_1_4"
colnames(covid)[colnames(covid) == "Lymphocytes < 1"] ="Lympho_lt_1"
colnames(covid)[colnames(covid) == "IL6 > 150"] ="IL6_gt_150"
colnames(covid)[colnames(covid) == "Ferritin > 300"] ="Ferritin_gt_300"
colnames(covid)[colnames(covid) == "C-Reactive Prot > 10"] ="CrctProtein_gt_10"
colnames(covid)[colnames(covid) == "Procalciton > 0.1"] ="Procalciton_gt_0"
colnames(covid)[colnames(covid) == "Troponin > 0.1"] ="Troponin_gt_0"

# Rearranging variables
covid <- covid[, c(2,1,3:61)]
#colnames(covid)
#covid

colnames(covid) 

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

round(covid1[1:3, c(1,25:33)], 2)

p_nonzero <- 10
tau0 <- p_nonzero/(p-p_nonzero) * 1/sqrt(n)
hs_prior <- hs(df=1, global_df=1, global_scale=tau0)
t_prior <- student_t(df = 7, location = 0, scale = 2.5)
fit <- stan_glm(y ~ LOS + Age_Range + Severity + Black + White + Asian + Latino +
                  MI + PVD + CHF + DEMENT + COPD + DM_C + DM_S + Renal + All_CNS + 
                  Pure_CNS + Stroke + Seizure + OldSyncope + OldOtherNeuro + 
                  OtherBrnLsn + Age + OsSats + OSat_lt_94 + Temp + Temp_gt_38 + 
                  MAP + MAP_lt_70 + Ddimer + Ddimer_gt_3 + Plts + INR + INR_gt_1.2 + 
                  BUN + BUN_gt_30 + Creatinine + Sodium + Sodium_bt_139_154 + 
                  Glucose + Glucose_bt_60_500 + AST + AST_gt_40 + ALT + ALT_gt_40 + 
                  WBC + WBC_bt_1_4 + Lympho + Lympho_lt_1 + IL6 + IL6_gt_150 + 
                  Ferritin + Ferritin_gt_300 + CrctProtein + CrctProtein_gt_10 + 
                  Procalcitonin + Procalciton_gt_0 + Troponin + Troponin_gt_0, 
                data = covid1, family=binomial(),
                prior = hs_prior, prior_intercept = t_prior, 
                seed = 1, adapt_delta = 0.99, refresh=0)


pplot <- plot(fit, "areas", prob = 0.95, prob_outer = 1)
plot1 <- pplot + geom_vline(xintercept = 0)
plot2 <- plot(fit)
grid.arrange(plot1, plot2, ncol=2)


fit_bayes <- stan_glm(y ~ Age + PVD + Renal + Stroke + OldSyncope + Temp + MAP + AST + 
                        Lympho + Ferritin_gt_300 + Procalcitonin + CrctProtein + Troponin, 
                      prior = normal(), prior_intercept = normal(), 
                      family=binomial(link="logit"),data= covid1)
summary(fit_bayes, digits= 3)

mcmc_dens(fit_bayes)

pp_check(fit_bayes, "dens_overlay")

p_direction(fit_bayes)


covid2 <- sample_n(covid1, 200)
fit1 <- stan_glm(y ~ PVD + Renal + Stroke + OldSyncope + OtherBrnLsn + Age + OSat_lt_94 + 
                   Temp + MAP + MAP_lt_70 + AST + Lympho_lt_1 + IL6_gt_150 + 
                   Ferritin_gt_300 + Procalcitonin + CrctProtein + 
                   Procalciton_gt_0 + Troponin, 
                 data = covid2, family=binomial(),
                 prior = hs_prior, prior_intercept = t_prior, 
                 seed = 1, adapt_delta = 0.99, refresh=0)
summary(fit1)[,1]
refmodel <- get_refmodel(fit1)
vs <- cv_varsel(refmodel, method='forward', cores=2)
plot(vs, stats = 'elpd')

plot(vs, stats = c('elpd', 'rmse'), deltas = TRUE)






