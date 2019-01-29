# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 17:44:12 2018

@author: xiong
"""
import os 
import pandas as pd
import numpy as np
os.chdir('E:\\Files\\Toronto Courses\\Risk Management\\Retail Risk Project')

var = pd.read_excel('MMF Data Dictionary.xlsx',0,header=0)
var = var.iloc[:,1:]

df = pd.read_csv('mmf_data2.csv',header=0,index_col=0)

##exclude widely held customer and deceased customers
temp =  (df['WIDELYHD']=='Y') | (df['deceased']==1)
df = df.loc[~temp]
df.drop(['WIDELYHD','deceased'],axis=1,inplace=True)

## default rate
y = df['t12']
y.sum() / len(y)

##creat explanatory variables
debit = ['debit_prev'+str(i) for i in range(1,13)]
debit.insert(0,'debit_curr')

credit = ['credit_prev'+str(i) for i in range(1,13)]
credit.insert(0,'credit_curr')

ratio = df[debit].values / df[credit].values
ratio_name = ['ratio_prev'+str(i) for i in range(1,13)]
ratio_name.insert(0,'ratio_curr')
ratio = pd.DataFrame(ratio,columns=ratio_name,index = df.index)

#debit.extend(credit)
#df.drop(columns = debit,axis=1,inplace=True)

df = pd.concat([df, ratio], axis=1, join='inner')


##delete other score
bench_name = ['benchmark'+str(i) for i in range(1,5)]
bench = df[bench_name]
df.drop(bench_name,axis=1,inplace=True)

## delet other target variable
temp = ['t'+str(i) for i in range(1,13)]
df.drop(temp,axis=1,inplace=True)

## delet those grouped vaiebles
variable = df.columns
grouped = []
for temp in variable:
    if temp.startswith(('WOE','GRP')):
        grouped.append(temp)
use_df = df[grouped]
df.drop(grouped,axis=1,inplace=True)

grouped = []
for temp in use_df.columns:
    if temp.startswith(('WOE')):
        grouped.append(temp)
use_df = use_df[grouped]


#%%Univariate screening
import copy 
dfc = copy.deepcopy(df)
##bin
import sklearn.tree as skt
def BinTree(x,y, groups = 5, depth = 5, min_impurity_decrease = 0):
    na_pos = pd.isna(x)
    xx = copy.deepcopy(x)
    if na_pos.sum()>0:
        nonna = xx[~na_pos]
        xx[na_pos] = np.nan ## to distinguish other categories
    else:
        nonna = xx
    tree = skt.DecisionTreeClassifier(max_depth = depth,max_features =None,class_weight='balanced',min_impurity_decrease = min_impurity_decrease,max_leaf_nodes=groups)
    tree.fit(nonna.values.reshape(-1,1),y[~na_pos])
    xx[~na_pos] = tree.apply(nonna.values.reshape(-1,1))
    return xx.values

def WOEIV(x,y):
    cate = pd.unique(x)
    total0 = (y==0).sum(); total1 = len(y) - total0
    WOE = []; sub0 = []; sub1 = [];
    for i,category in enumerate(cate):
        if type(category) != np.str:
            if pd.isna(category):
                sub0.append((y[pd.isna(x)]==0).sum()); sub1.append((y[pd.isna(x)]==1).sum())
            else:
                sub0.append((y[x==category]==0).sum()); sub1.append((y[x==category]==1).sum())
        else:
            sub0.append((y[x==category]==0).sum()); sub1.append((y[x==category]==1).sum())
        
        if sub0[i]==0:
            WOE.append(np.log(sub1[i] / total1))
        elif sub1[i]==0:
            WOE.append(-np.log(sub0[i] / total0))
        else:
            WOE.append(np.log(sub0[i] / total0) - np.log(sub1[i] / total1))
    WOE = np.array(WOE)
    sub0 = np.array(sub0); sub1 = np.array(sub1)
    return [WOE,(sub0/total0 - sub1/total1) * WOE,np.sum((sub0/total0 - sub1/total1) * WOE)]
    

for i in np.arange(13,0,-1):
    x = dfc.iloc[:,-i].values.reshape(-1,1)
    dfc.iloc[:,-i] = BinTree(x,y)
    x = dfc.iloc[:,-i].values
    print('--ratio Previous--'+str(13-i),round(WOEIV(x,y)[1],4),sep = '\t')
    ww = WOEIV(x,y)[0]
    '''
    for j,cate in enumerate(np.unique(dfc.iloc[:,-i].values)):
        temp = dfc.iloc[:,-i] == cate
        dfc.ix[temp,-i] = ww[j]
    '''
#%%
def IV_given_WOE(x,y):
    cate = np.unique(x)
    total0 = (y==0).sum(); total1 = len(y) - total0
    score = 0
    for kind in cate:
        sub0 = (y[x==kind]==0).sum()
        sub1 = (y[x==kind]==1).sum()
        score += (sub0/total0 - sub1/total1) * kind
    return score
##
grouped_iv = []
for name in use_df.columns:
    x = use_df[name]
    grouped_iv.append(IV_given_WOE(x,y))
    print(name, IV_given_WOE(x,y),sep = '\t')
np.min(grouped_iv)
  
##drop those variables with nan more than 99%
temp = dfc.columns[pd.isna(dfc).sum()/len(dfc) > 0.995]
dfc.drop(temp,axis=1,inplace=True)

## compute IV for all left variables
iv = []
for i,name in enumerate(dfc.columns):
    x = dfc[name]
    if len(pd.unique(x)) >10:
        x = BinTree(x,y,10)
    print(name, WOEIV(x,y)[2],sep = '\t')
    iv.append(WOEIV(x,y)[2])

dfc = dfc.loc[:,np.array(iv)>0.1]


def cate_summary(x,y):
    for i,cate in enumerate(pd.unique(x)):
        print('NonDefault ', (y[x==cate]==0).sum())
        print('Default ', (y[x==cate]==1).sum())
        print('Default rate ', (y[x==cate]==1).sum() / ((y[x==cate]==1).sum() +(y[x==cate]==0).sum()) )
        
##pick high and low IV
low_name = 'SPP_Group_1'
x = dfc[low_name]
WOEIV(x,y)[1]

high_name = 'TBSSC100'
x = dfc[high_name]

## change all data points
for i,name in enumerate(dfc.columns):
    x = dfc[name]
    if len(pd.unique(x)) >10:
        x = BinTree(x,y,10)
    ww = WOEIV(x,y)[0]
    for j,cate in enumerate(pd.unique(x)):
        temp = (x == cate) | (pd.isna(x))
        dfc.loc[temp,name] = ww[j]
  
#%%
from var_clus import VarClus

demo = VarClus(max_eigenvalue=1.35,max_tries = 5)
demo.decompose(dfc)
demo.print_cluster_structure()

#%%stepwise
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(class_weight='balanced',random_state=11,solver = 'lbfgs')
rfe = RFE(logreg, 44)
rfe.fit(dfc,y)
rfe.predict_proba(dfc)

selectX = rfe.transform(dfc)
## find those selected variables
for i in range(44):
    temp = selectX[:,i]
    for name in dfc.columns:
        temp1 = dfc[name]
        if (temp1 == temp).all():
            print(name)

prob = rfe.predict_proba(dfc)
odds = prob[:,0] / prob[:,1]



#%%
import matplotlib.pyplot as plt
predicty = rfe.predict(dfc)
score = 633.56 + np.log(odds) * 20 / np.log(2)
cut = [-np.inf, 560,600,640,680,720,740,780,820,np.inf]
category  =  pd.cut(score,cut,labels = np.arange(9))

def cate_no(category, y, default = True):
    total1 = (y==1).sum(); total0 = (y==0).sum()
    temp = []
    if default:
        for cate in np.sort(pd.unique(category)):
            temp.append( (y[category==cate]==1).sum()/total1 )
    else:
        for cate in np.sort(pd.unique(category)):
            temp.append( (y[category==cate]==0).sum()/total0 )
    return np.array(temp)


def KS(category,y):
    cum_default = cate_no(category,y).cumsum()
    cum_nondefault = cate_no(category,y,False).cumsum()
    #xtick = ['<560', '560-600','600-640','640-680','680-720','720-760','760-780','780-820','>820']
    xtick = [ '560-600','600-640','640-680','680-720','720-760','760-780','780-820','>820']
    plt.plot(xtick,cum_default,'k-',label = 'Default')
    plt.plot(xtick,cum_nondefault,'r--',label = 'Non-Default')
    plt.legend()
    plt.title('KS = '+ str(round(max(cum_default - cum_nondefault),4)))

KS(category,y)

## ar curve
capcurve(y, predicty)

## population stability
from collections import Counter
time = df['TIME_KEY']
unique_time = np.sort(pd.unique(time))
def Population_sta(score,which):
    score1 = score[time == unique_time[which]]
    cate1 = pd.cut(score1,cut,labels = np.arange(9))
    percent = []
    for i in np.arange(9):
        percent.append((cate1 ==i).sum()/ len(score1) )
    return np.array(percent)

xxtick = np.arange(1,10)
Population_sta(score,1)
plt.bar(xxtick-0.3,Population_sta(score,0),width = 0.15, label = 'Jan 2014')
plt.bar(xxtick-0.15,Population_sta(score,1),width = 0.15, label = 'Apr 2014')
plt.bar(xxtick,Population_sta(score,2),width = 0.15, label = 'July 2014')
plt.bar(xxtick+0.15,Population_sta(score,2),width = 0.15, label = 'Nov 2014')
plt.legend()
plt.xlabel('Score Group')



## bench comparison
pd.isna(bench).sum()
benchmark4 = bench.iloc[:,3]
category4 = pd.cut(benchmark4, cut,labels = np.arange(9))
KS(category4,y)

benchmark1 = bench.iloc[:,0]
category1 = pd.cut(benchmark1, cut,labels = np.arange(9))
KS(category1,y)

##
predicty1 = np.where(benchmark1>633.56,0,1)
capcurve(y, predicty1)
##
predicty4 = np.where(benchmark4>633.56,0,1)
capcurve(y, predicty4)
