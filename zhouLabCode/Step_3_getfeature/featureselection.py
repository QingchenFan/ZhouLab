import scipy.io as scio
import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import rankdata
'''
    Feature Selection
'''

def fdr(p_vals):


    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr
gradientdata = np.loadtxt('/Users/fan/Documents/Data/zhouTestData/featureFC/IPCAS_GordonsubcorticalFC.txt')

behaviordata = pd.read_csv('/Users/fan/Documents/Data/zhouTestData/beh_info.csv')
bedata = behaviordata['BIS']
bedata = np.array(bedata)
bedata_2 = behaviordata['TAI']
bedata_3 = behaviordata['Reflection']
correlation, pvalue = scipy.stats.pearsonr(bedata_3, bedata_2)
print('correlation:', correlation, '-pvalue:', pvalue)

corr = []
pv = []
res = []
num = 0
for i in range(0, gradientdata.shape[1]):
    temp = gradientdata[:, i]
    correlation, pvalue = scipy.stats.pearsonr(temp, bedata)
    #print('--correlation--', correlation, '--pvalue--', pvalue)
    if pvalue <= 0.05:
        #num = num + 1
        print('--correlation--', correlation, '--pvalue--', pvalue)
    pv.append(pvalue)
    corr.append(correlation)

pvarr = np.array(pv)

print('pvarr-', pvarr)
fdrres_1 = multipletests(pvarr, method='fdr_bh', alpha=0.05, is_sorted=False)
fdrres_2 = fdr(pvarr)
print('fdrres_1-', fdrres_1)
print(fdrres_1[1])
print('fdrres_2-', fdrres_2)
for i in fdrres_2:
    print(i)
    if i <=0.05:
        num = num + 1
        print('i-',i)
# print(num)
# preTrueCovari = pd.DataFrame(corr)
# pvcsv = pd.DataFrame(pv)
# res.append(preTrueCovari)
# res.append(pvcsv)
# datares = pd.concat(res, axis=1)
# datares.to_csv('./BIScorrfeature.csv')
