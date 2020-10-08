import pandas as pd
import numpy as np
import scipy 
from scipy import stats
import statsmodels.api as sm
import statsmodels.stats.descriptivestats as sd
 import plotly as py
 import plotly.graph_objs as go
 from plotly.tools import FigureFactory as FF
from plotly import figure_factory as FF





############2 sample T Test(Marketing Strategy) ##################

#promotion=pd.read_excel("C:\Datasets_BA\Hypothesis\Hypothesis Testing_R&Python_codes\Promotion.xlsx")

promotion=pd.read_excel("C:\\Users\\nitin\\Downloads\\Python_Unsupervised\\Python_Unsupervised\\Hypothesis Testing\\Promotion.xlsx")
promotion
promotion.columns

######## 2 Sample T test ################
scipy.stats.ttest_ind(promotion.InterestRateWaiver,promotion.StandardPromotion)
help(scipy.stats.ttest_ind)


############# One - Way Anova###################
from statsmodels.formula.api import ols

cof=pd.read_excel("C:\\Users\\nitin\\Downloads\\Python_Unsupervised\\Python_Unsupervised\\Hypothesis Testing\\ContractRenewal_Data(unstacked).xlsx")
cof
cof.columns="SupplierA","SupplierB","SupplierC"

##########Normality Test ############

print(stats.shapiro(cof.SupplierA))    #Shapiro Test
print(stats.shapiro(cof.SupplierB))
print(stats.shapiro(cof.SupplierC))

############## Variance test #########
scipy.stats.levene(cof.SupplierA, cof.SupplierB)
scipy.stats.levene(cof.SupplierB, cof.SupplierC)
scipy.stats.levene(cof.SupplierC, cof.SupplierA)

############# One - Way Anova###################
mod= ols('SupplierA ~ C(SupplierB, SupplierC)',data=cof).fit()

aov_table=sm.stats.anova_lm(mod, type=2)
help(sm.stats.anova_lm)

print(aov_table)



