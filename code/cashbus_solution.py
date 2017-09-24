#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#notice the default cases are labeled as 0 while non-default cases have value of 1
#among the 15k records there are 1542 cases have label of 0 and 13458 have label of 1
#the unbalance rate is 10%

#according to offical doc, all records have negative values are actually missing values


#set working directory before excuting the following code

################################
#       1.missing values       #
################################

#count na by row(the value is -1 as noticed in the offical doc) 
#discretize the row-wise missing value count into 5 levels

import pandas as pd
train_x = pd.read_csv("train_x.csv")
train_x['uid'].to_csv("uid.csv")
test_x = pd.read_csv("test_x.csv")

#count missing values per row
train_x['n_null'] = (train_x<0).sum(axis=1)
test_x['n_null'] = (test_x<0).sum(axis=1)
train_x.shape
list(train_x)

#do binning to the row-wise missing value counts
train_x['discret_null'] = train_x.n_null
train_x.discret_null[train_x.discret_null<=32] = 1
train_x.discret_null[(train_x.discret_null>32)&(train_x.discret_null<=69)] = 2
train_x.discret_null[(train_x.discret_null>69)&(train_x.discret_null<=147)] = 3
train_x.discret_null[(train_x.discret_null>147)&(train_x.discret_null<=194)] = 4
train_x.discret_null[(train_x.discret_null>194)] = 5
train_x[['uid','n_null','discret_null']].to_csv('train_x_null.csv',index=None)

test_x['discret_null'] = test_x.n_null
test_x.discret_null[test_x.discret_null<=32] = 1
test_x.discret_null[(test_x.discret_null>32)&(test_x.discret_null<=69)] = 2
test_x.discret_null[(test_x.discret_null>69)&(test_x.discret_null<=147)] = 3
test_x.discret_null[(test_x.discret_null>147)&(test_x.discret_null<=194)] = 4
test_x.discret_null[(test_x.discret_null>194)] = 5
test_x[['uid','n_null','discret_null']].to_csv('test_x_null.csv',index=None)


################################
#   2.num variable rankings    #
################################

#deal with ranking of numeric feature value
#create ranking of each cell in its column

feature_type = pd.read_csv('features_type.csv')
numeric_feature = list(feature_type[feature_type.type=='numeric'].feature)

#naming rules for rank features：add "r" in front of original featuer name, for instance rank feature for "x1" is "rx1"

#do ranking for three files seperately, normalization is in need 
#it makes more sense to merge train and test files and do overal ranking 
#the result is found to be similar to this one after a trial

test = pd.read_csv('test_x.csv')[['uid']+numeric_feature]#numeric only
test_rank = pd.DataFrame(test.uid,columns=['uid'])
for feature in numeric_feature:
    test_rank['r'+feature] = test[feature].rank(method='max')
test_rank.to_csv('test_x_rank.csv',index=None)


train = pd.read_csv('train_x.csv')[['uid']+numeric_feature]
train_rank = pd.DataFrame(train.uid,columns=['uid'])
for feature in numeric_feature:
    train_rank['r'+feature] = train[feature].rank(method='max')
train_rank.to_csv('train_x_rank.csv',index=None)


train_unlabeled = pd.read_csv('train_unlabeled.csv')[['uid']+numeric_feature]
train_unlabeled_rank = pd.DataFrame(train_unlabeled.uid,columns=['uid'])
for feature in numeric_feature:
    train_unlabeled_rank['r'+feature] = train_unlabeled[feature].rank(method='max')
train_unlabeled_rank.to_csv('train_unlabeled_rank.csv',index=None)

#####################################
#   3.discretization of rankings    #
#####################################
#create discretization features

train = pd.read_csv("train_x_rank.csv")
train_x = train.drop(['uid'],axis=1)
test = pd.read_csv("test_x_rank.csv")
test_x = test.drop(['uid'],axis=1)
train_unlabeled =  pd.read_csv("train_unlabeled_rank.csv")
train_unlabeled_x =  train_unlabeled.drop(['uid'],axis=1)

#discretization of ranking features
#each 10% belongs to 1 level
train_x[train_x<1500] = 1
train_x[(train_x>=1500)&(train_x<3000)] = 2
train_x[(train_x>=3000)&(train_x<4500)] = 3
train_x[(train_x>=4500)&(train_x<6000)] = 4
train_x[(train_x>=6000)&(train_x<7500)] = 5
train_x[(train_x>=7500)&(train_x<9000)] = 6
train_x[(train_x>=9000)&(train_x<10500)] = 7
train_x[(train_x>=10500)&(train_x<12000)] = 8
train_x[(train_x>=12000)&(train_x<13500)] = 9
train_x[train_x>=13500] = 10
       
#nameing rule for discretization features, add "d" in front of orginal features
#for instance "x1" would have discretization feature of "dx1"
rename_dict = {s:'d'+s[1:] for s in train_x.columns.tolist()}
train_x = train_x.rename(columns=rename_dict)
train_x['uid'] = train.uid
train_x.to_csv('train_x_discretization.csv',index=None)


train_unlabeled_x[train_unlabeled_x<5000] = 1
train_unlabeled_x[(train_unlabeled_x>=5000)&(train_unlabeled_x<10000)] = 2
train_unlabeled_x[(train_unlabeled_x>=10000)&(train_unlabeled_x<15000)] = 3
train_unlabeled_x[(train_unlabeled_x>=15000)&(train_unlabeled_x<20000)] = 4
train_unlabeled_x[(train_unlabeled_x>=20000)&(train_unlabeled_x<25000)] = 5
train_unlabeled_x[(train_unlabeled_x>=25000)&(train_unlabeled_x<30000)] = 6
train_unlabeled_x[(train_unlabeled_x>=30000)&(train_unlabeled_x<35000)] = 7
train_unlabeled_x[(train_unlabeled_x>=35000)&(train_unlabeled_x<40000)] = 8
train_unlabeled_x[(train_unlabeled_x>=40000)&(train_unlabeled_x<45000)] = 9
train_unlabeled_x[train_unlabeled_x>=45000] = 10
train_unlabeled_x = train_unlabeled_x.rename(columns=rename_dict)
train_unlabeled_x['uid'] = train_unlabeled.uid
train_unlabeled_x.to_csv('train_unlabeled_discretization.csv',index=None)

test_x[test_x<500] = 1
test_x[(test_x>=500)&(test_x<1000)] = 2
test_x[(test_x>=1000)&(test_x<1500)] = 3
test_x[(test_x>=1500)&(test_x<2000)] = 4
test_x[(test_x>=2000)&(test_x<2500)] = 5
test_x[(test_x>=2500)&(test_x<3000)] = 6
test_x[(test_x>=3000)&(test_x<3500)] = 7
test_x[(test_x>=3500)&(test_x<4000)] = 8
test_x[(test_x>=4000)&(test_x<4500)] = 9
test_x[test_x>=4500] = 10
test_x = test_x.rename(columns=rename_dict)
test_x['uid'] = test.uid
test_x.to_csv('test_x_discretization.csv',index=None)

#############################################
#   4.frequency of ranking discretization   #
#############################################

#count of discretization of rankings

train_x = pd.read_csv('train_x_discretization.csv')
test_x = pd.read_csv('test_x_discretization.csv')
train_unlabeled_x =  pd.read_csv('train_unlabeled_discretization.csv')

train_x['n1'] = (train_x==1).sum(axis=1)
train_x['n2'] = (train_x==2).sum(axis=1)
train_x['n3'] = (train_x==3).sum(axis=1)
train_x['n4'] = (train_x==4).sum(axis=1)
train_x['n5'] = (train_x==5).sum(axis=1)
train_x['n6'] = (train_x==6).sum(axis=1)
train_x['n7'] = (train_x==7).sum(axis=1)
train_x['n8'] = (train_x==8).sum(axis=1)
train_x['n9'] = (train_x==9).sum(axis=1)
train_x['n10'] = (train_x==10).sum(axis=1)
train_x[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']].to_csv('train_x_nd.csv',index=None)

test_x['n1'] = (test_x==1).sum(axis=1)
test_x['n2'] = (test_x==2).sum(axis=1)
test_x['n3'] = (test_x==3).sum(axis=1)
test_x['n4'] = (test_x==4).sum(axis=1)
test_x['n5'] = (test_x==5).sum(axis=1)
test_x['n6'] = (test_x==6).sum(axis=1)
test_x['n7'] = (test_x==7).sum(axis=1)
test_x['n8'] = (test_x==8).sum(axis=1)
test_x['n9'] = (test_x==9).sum(axis=1)
test_x['n10'] = (test_x==10).sum(axis=1)
test_x[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']].to_csv('test_x_nd.csv',index=None)

train_unlabeled_x['n1'] = (train_unlabeled_x==1).sum(axis=1)
train_unlabeled_x['n2'] = (train_unlabeled_x==2).sum(axis=1)
train_unlabeled_x['n3'] = (train_unlabeled_x==3).sum(axis=1)
train_unlabeled_x['n4'] = (train_unlabeled_x==4).sum(axis=1)
train_unlabeled_x['n5'] = (train_unlabeled_x==5).sum(axis=1)
train_unlabeled_x['n6'] = (train_unlabeled_x==6).sum(axis=1)
train_unlabeled_x['n7'] = (train_unlabeled_x==7).sum(axis=1)
train_unlabeled_x['n8'] = (train_unlabeled_x==8).sum(axis=1)
train_unlabeled_x['n9'] = (train_unlabeled_x==9).sum(axis=1)
train_unlabeled_x['n10'] = (train_unlabeled_x==10).sum(axis=1)
train_unlabeled_x[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']].to_csv('train_unlabeled_nd.csv',index=None)


##############################################
#   5.feature importance of rank features    #
##############################################
#generate a variety of xgboost models to have rank feature importance

import pandas as pd
import xgboost as xgb
import sys
import random
import _pickle as cPickle
import os

#craete a dicrectory if it doesn't exist
#https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
if not os.path.exists("featurescore"):
    os.makedirs("featurescore")
if not os.path.exists("model"):
    os.makedirs("model")    
if not os.path.exists("preds"):
    os.makedirs("preds")    


#load data
train_x = pd.read_csv("train_x_rank.csv")
train_y = pd.read_csv("train_y.csv")
train_xy = pd.merge(train_x,train_y,on='uid')
y = train_xy.y

#leave features only
train_x= train_xy.drop(["uid",'y'],axis=1)
#convert to percentage 
X = train_x/15000.0
#to xgb.DMatrix format
dtrain = xgb.DMatrix(X, label=y)

#do the same to test table    
test = pd.read_csv("test_x_rank.csv")
test_uid = test.uid
test = test.drop("uid",axis=1)
test_x = test/5000.0
dtest = xgb.DMatrix(test_x)

#define an xgb model to do calculate feature importance 
def pipeline(iteration,random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight):
    params={
    	'booster':'gbtree',
    	'objective': 'binary:logistic',
    	'scale_pos_weight': float(len(y)-sum(y))/float(sum(y)),
        'eval_metric': 'auc',
    	'gamma':gamma,
    	'max_depth':max_depth,
    	'lambda':lambd,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight, 
        'eta': 0.04,
    	'seed':random_seed,
        }
    
    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=1350,evals=watchlist)
    model.save_model('./model/xgb{0}.model'.format(iteration))
    
    #predict test set
    test_y = model.predict(dtest)
    test_result = pd.DataFrame(columns=["uid","score"])
    test_result.uid = test_uid
    test_result.score = test_y
    test_result.to_csv("./preds/xgb{0}.csv".format(iteration),index=None,encoding='utf-8')
    
    #save feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    
    with open('./featurescore/feature_score_{0}.csv'.format(iteration),'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)



if __name__ == "__main__":
    random_seed = list(range(1000,2000,10))
    gamma = [i/1000.0 for i in list(range(100,200,1))]
    max_depth = [6,7,8]
    lambd = list(range(100,200,1))
    subsample = [i/1000.0 for i in list(range(500,700,2))]
    colsample_bytree = [i/1000.0 for i in list(range(250,350,1))]
    min_child_weight = [i/1000.0 for i in list(range(200,300,1))]
    random.shuffle(random_seed)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    
#save params for reproducing
    with open('params.pkl','wb') as f:
        cPickle.dump((random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight),f)
    

#train 100 xgb
    for i in list(range(100)):
        pipeline(i,random_seed[i],gamma[i],max_depth[i%3],lambd[i],subsample[i],colsample_bytree[i],min_child_weight[i])


#calculate average feature score for ranking features

#get rank feature importance info from the xgboost models
import pandas as pd 
import os

#featurescore folder contains csv files called feature_score_* that tells feature importance

files = os.listdir('featurescore')
#save into a dict
fs = {}
for f in files:
    t = pd.read_csv('featurescore/'+f)
    t.index = t.feature
    t = t.drop(['feature'],axis=1)
    d = t.to_dict()['score']
    for key in d:
        if key in fs:
            fs[key] += d[key]
        else:
            fs[key] = d[key] 
       
#sort and organize the dict            
fs = sorted(fs.items(), key=lambda x:x[1],reverse=True)

t = []
type(t)
for (key,value) in fs:
    t.append("{0},{1}\n".format(key,value))

#save the overall importance scores of ranking features into csv
with open('rank_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(t)


##############################################
#     6.feature importance of raw features   #
##############################################
#generate a variety of xgboost models to have raw feature importance

print(__doc__)

from sklearn.cross_validation import train_test_split
import pandas as pd
import xgboost as xgb

random_seed = 1225

#set data path
train_x_csv = "train_x.csv"
train_y_csv = "train_y.csv"
test_x_csv = "test_x.csv"
features_type_csv = "features_type.csv"

#load data
train_x = pd.read_csv(train_x_csv)
train_y = pd.read_csv(train_y_csv)
train_xy = pd.merge(train_x,train_y,on='uid')

test = pd.read_csv(test_x_csv)
test_uid = test.uid
test_x = test.drop(['uid'],axis=1)

#dictionary {feature:type}
features_type = pd.read_csv(features_type_csv)
features_type.index = features_type.feature
features_type = features_type.drop('feature',axis=1)
features_type = features_type.to_dict()['type']


feature_info = {}
features = list(train_x.columns)
features.remove('uid')

for feature in features:
    max_ = train_x[feature].max()
    min_ = train_x[feature].min()
    n_null = len(train_x[train_x[feature]<0])  #number of null
    n_gt1w = len(train_x[train_x[feature]>10000])  #greater than 10000
    feature_info[feature] = [min_,max_,n_null,n_gt1w]

#see how many neg/pos sample 10%
print(len(train_xy[train_xy.y==0])/(len(train_xy[train_xy.y==1])+len(train_xy[train_xy.y==0])))

#split train set,generate train,val,test set
train_xy = train_xy.drop(['uid'],axis=1)
#80% of whole training dataset is left for training while 20% is used for validation
train,val = train_test_split(train_xy, test_size = 0.2,random_state=1)
#train label
y = train.y
#train x
X = train.drop(['y'],axis=1)
#validation y
val_y = val.y
#validaton x
val_X = val.drop(['y'],axis=1)


#convert  dataset to xgb.DMatrix
dtest = xgb.DMatrix(test_x)
dval = xgb.DMatrix(val_X,label=val_y)
dtrain = xgb.DMatrix(X, label=y)


#parameters for training xgb
params={
	'booster':'gbtree',
	'objective': 'binary:logistic',
	'early_stopping_rounds':100,
	'scale_pos_weight': 1500.0/13458.0,
        'eval_metric': 'auc',
	'gamma':0.1,
	'max_depth':8,
	'lambda':550,
        'subsample':0.7,
        'colsample_bytree':0.3,
        'min_child_weight':2.5, 
        'eta': 0.007,
	'seed':random_seed,
	'nthread':7
    }

watchlist  = [(dtrain,'train'),(dval,'val')]
model = xgb.train(params,dtrain,num_boost_round=5000,evals=watchlist)
#save model
model.save_model('./model/xgb.model')
#save best
print ("best best_ntree_limit",model.best_ntree_limit )  

#predict test set (from the best iteration)
test_y = model.predict(dtest,ntree_limit=model.best_ntree_limit)
test_result = pd.DataFrame(columns=["uid","score"])
test_result.uid = test_uid
test_result.score = test_y
test_result.to_csv("xgb.csv",index=None,encoding='utf-8')


#calculate average feature score

#save feature score and feature information:  feature,score,min,max,n_null,n_gt1w
feature_score = model.get_fscore()
for key in feature_score:
    feature_score[key] = [feature_score[key]]+feature_info[key]+[features_type[key]]

feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
fs = []
for (key,value) in feature_score:
    fs.append("{0},{1},{2},{3},{4},{5},{6}\n".format(key,value[0],value[1],value[2],value[3],value[4],value[5]))

with open('feature_score.csv','w') as f:
    f.writelines("feature,score,min,max,n_null,n_gt1w\n")
    f.writelines(fs)

##################################################
#     7.feature importance of discret features   #
##################################################

#generate a variety of xgboost models to have discret feature importance
from sklearn.cross_validation import train_test_split
import pandas as pd
import xgboost as xgb
import sys,random
import _pickle as cPickle
import os

if not os.path.exists("featurescore"):
    os.makedirs("featurescore")
if not os.path.exists("model"):
    os.makedirs("model")    
if not os.path.exists("preds"):
    os.makedirs("preds") 


#use discret feature 'dx*' and 'n1'~'n10'
#load data
train_x_d = pd.read_csv('train_x_discretization.csv')
train_x_nd = pd.read_csv('train_x_nd.csv')
train_x = pd.merge(train_x_d,train_x_nd,on='uid')
train_y = pd.read_csv('train_y.csv')
train_xy = pd.merge(train_x,train_y,on='uid')
y = train_xy.y
X = train_xy.drop(['uid','y'],axis=1)
#to xgb.DMatrix format
dtrain = xgb.DMatrix(X, label=y)

#do the same to test table  
test_x_d = pd.read_csv('test_x_discretization.csv')
test_x_nd = pd.read_csv('test_x_nd.csv')
test = pd.merge(test_x_d,test_x_nd,on='uid')
test_uid = test.uid
test_x = test.drop("uid",axis=1)
dtest = xgb.DMatrix(test_x)


def pipeline(iteration,random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight):
    if max_depth==6:
        num_boost_round = 550
    elif max_depth==7:
        num_boost_round = 450
    elif max_depth==8:
        num_boost_round = 400
    
    params={
    	'booster':'gbtree',
    	'objective': 'rank:pairwise',
    	'scale_pos_weight': float(len(y)-sum(y))/float(sum(y)),
        'eval_metric': 'auc',
    	'gamma':gamma,
    	'max_depth':max_depth,
    	'lambda':lambd,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight, 
        'eta': 0.04,
    	'seed':random_seed
        }

    
    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=num_boost_round,evals=watchlist)
    model.save_model('./model/xgb{0}.model'.format(iteration))
    
    #predict test set
    test_y = model.predict(dtest)
    test_result = pd.DataFrame(test_uid,columns=["uid"])
    test_result['score'] = test_y
    test_result.to_csv("./preds/xgb{0}.csv".format(iteration),index=None,encoding='utf-8')
    
    #get feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    
    with open('./featurescore/feature_score_{0}.csv'.format(iteration),'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)



if __name__ == "__main__":
    random_seed = list(range(1000,2000,20))
    gamma = [i/1000.0 for i in range(100,200,2)]
    max_depth = [6,7,8]
    lambd = list(range(200,400,2))
    subsample = [i/1000.0 for i in list(range(600,700,2))]
    colsample_bytree = [i/1000.0 for i in list(range(250,350,2))]
    min_child_weight = [i/1000.0 for i in list(range(200,300,2))]
    random.shuffle(random_seed)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    
    with open('params.pkl','wb') as f:
        cPickle.dump((random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight),f)

    
    for i in list(range(36)):
        pipeline(i,random_seed[i],gamma[i],max_depth[i%3],lambd[i],subsample[i],colsample_bytree[i],min_child_weight[i])

#calculate average feature score
#feature importance of discret features from the xgb modles above
import pandas as pd 
import os


files = os.listdir('featurescore')
fs = {}
for f in files:
    t = pd.read_csv('featurescore/'+f)
    t.index = t.feature
    t = t.drop(['feature'],axis=1)
    d = t.to_dict()['score']
    for key in d:
        if key in ['n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']:
            continue
        if key in fs:
            fs[key] += d[key]
        else:
            fs[key] = d[key] 
            
 #sort and organize the dict                       
fs = sorted(fs.items(), key=lambda x:x[1],reverse=True)

t = []
for (key,value) in fs:
    t.append("{0},{1}\n".format(key,value))
    
#save the overall importance scores of discret features into csv
with open('discret_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(t)

######################################################################
#          8.feature selection based on feature importance           #
######################################################################    

if not os.path.exists("featurescore"):
    os.makedirs("featurescore")
if not os.path.exists("model"):
    os.makedirs("model")    
if not os.path.exists("preds"):
    os.makedirs("preds") 

#count features of ranking discretion
test_nd = pd.read_csv('test_x_nd.csv')[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']]
train_nd = pd.read_csv('train_x_nd.csv')[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']]
trainunlabeled_nd = pd.read_csv('train_unlabeled_nd.csv')[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']]

#discret features of count of missing values
test_dnull = pd.read_csv('test_x_null.csv')[['uid','discret_null']]
train_dnull = pd.read_csv('train_x_null.csv')[['uid','discret_null']]
trainunlabeled_dnull = pd.read_csv('train_unlabeled_null.csv')[['uid','discret_null']]

#considering the size of the features above (only 11) it is not necessary to do feature selection on them
#so they are merged and left alone in the feature selection process
eleven_feature = ['n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','discret_null']
test_eleven = pd.merge(test_nd,test_dnull,on='uid')
train_eleven = pd.merge(train_nd,train_dnull,on='uid')
trainunlabeled_eleven = pd.merge(trainunlabeled_nd,trainunlabeled_dnull,on='uid')

del test_dnull,train_dnull,trainunlabeled_dnull
del test_nd,train_nd,trainunlabeled_nd


#discret features

#this file is already ordered by the feature importance, the more important the feature is, the higher the ranking it has
discret_feature_score = pd.read_csv('discret_feature_score.csv')
#save the top 500 feature names into a list
fs = list(discret_feature_score.feature[0:500])
#discret features with top 500 important features only
discret_train = pd.read_csv("train_x_discretization.csv")[['uid']+fs]
discret_test = pd.read_csv("test_x_discretization.csv")[['uid']+fs]
discret_train_unlabeled = pd.read_csv("train_unlabeled_discretization.csv")[['uid']+fs]

#ranking features
rank_feature_score = pd.read_csv('rank_feature_score.csv')
fs = list(rank_feature_score.feature[0:500])
rank_train_x = pd.read_csv("train_x_rank.csv")
#select top 500 most important ranking features and divided by 15,000 to gain the percentage vale
rank_train = rank_train_x[fs] / float(len(rank_train_x))
#add uid
rank_train['uid'] = rank_train_x.uid

rank_test_x = pd.read_csv("test_x_rank.csv")
rank_test = rank_test_x[fs] / float(len(rank_test_x))
rank_test['uid'] = rank_test_x.uid

rank_train_unlabeled_x = pd.read_csv("train_unlabeled_rank.csv")
rank_train_unlabeled = rank_train_unlabeled_x[fs] / float(len(rank_train_unlabeled_x))
rank_train_unlabeled['uid'] = rank_train_unlabeled_x.uid

del rank_train_x,rank_test_x,rank_train_unlabeled_x

#raw features
raw_feature_score = pd.read_csv('raw_feature_score.csv')
fs = list(raw_feature_score.feature[0:500])
raw_train_x = pd.read_csv("train_x.csv")[['uid']+fs]
raw_train_y = pd.read_csv("train_y.csv")
raw_train = pd.merge(raw_train_x,raw_train_y,on='uid')
del raw_train_x,raw_train_y

raw_test = pd.read_csv("test_x.csv")[['uid']+fs]
raw_train_unlabel = pd.read_csv('train_unlabeled.csv')[['uid']+fs]

#merge raw, ranking, discret and other 11 features
train = pd.merge(raw_train,rank_train,on='uid')
train = pd.merge(train,discret_train,on='uid')
train = pd.merge(train,train_eleven,on='uid')
#train 15,000 * 1,513
test = pd.merge(raw_test,rank_test,on='uid')
test = pd.merge(test,discret_test,on='uid')
test = pd.merge(test,test_eleven,on='uid')
test_uid = test.uid
#train 5,000 * 1,512


#remove all samples have missing value discret level 5
#in another word those have more than 194 missing values
train = train[train.discret_null!=5]
#thus 100 records are removed

    ##################################
    #       save prepared data       #
    ##################################

#save as csv, label is called target
#rename label from y to target
train.rename(columns={'y':'target'}, inplace=True)
train.to_csv('cashbus_train_withid.csv',index=None)

train_noid=train.drop(['uid'],axis=1)
train_noid.to_csv('cashbus_train_noid.csv',index=None)



#feature selection in a bagging manner
#create randomness in the number of raw,ranking and discret features 
#by setting the number of feature from a random number from 300 to 500
#feature_num is such a varaible


def pipeline(iteration,random_seed,feature_num,rank_feature_num,discret_feature_num,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight):
    #define number of features as a variable feature_num
    raw_feature_selected = list(raw_feature_score.feature[0:feature_num])
    rank_feature_selected = list(rank_feature_score.feature[0:rank_feature_num])
    discret_feature_selected = list(discret_feature_score.feature[0:discret_feature_num])

    #construct training dataset from the randomly selected top features from raw, ranking, discret plus untouched 11
    train_xy = train[eleven_feature+raw_feature_selected+rank_feature_selected+discret_feature_selected+['y']]
    #unify all missing records to -1
    train_xy[train_xy<0] = -1

    test_x = test[eleven_feature+raw_feature_selected+rank_feature_selected+discret_feature_selected]
    test_x[test_x<0] = -1

    y = train_xy.y
    X = train_xy.drop(['y'],axis=1)
    
    ####################
    #       xgb        #
    ####################
    dtest = xgb.DMatrix(test_x)
    dtrain = xgb.DMatrix(X, label=y)
    
    params={
    	'booster':'gbtree',
    	'objective': 'binary:logistic',
    	'scale_pos_weight': float(len(y)-sum(y))/float(sum(y)),
        'eval_metric': 'auc',
    	'gamma':gamma,
    	'max_depth':max_depth,
    	'lambda':lambd,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight, 
        'eta': 0.08,
    	'seed':random_seed,
    	'nthread':8
        }
    
    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=1500,evals=watchlist)
    model.save_model('./model/xgb{0}.model'.format(iteration))
    
    #predict test set
    test_y = model.predict(dtest)
    test_result = pd.DataFrame(test_uid,columns=["uid"])
    test_result["score"] = test_y
    test_result.to_csv("./preds/xgb{0}.csv".format(iteration),index=None,encoding='utf-8')
    
    #save feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    
    with open('./featurescore/feature_score_{0}.csv'.format(iteration),'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)



if __name__ == "__main__":
    
    
    random_seed = list(range(1000,2000,10))
    feature_num = list(range(300,500,2))
    rank_feature_num = list(range(300,500,2))
    discret_feature_num = list(range(64,100,1))
    gamma = [i/1000.0 for i in list(range(0,300,3))]
    max_depth = [6,7,8]
    lambd = list(range(500,700,2))
    subsample = [i/1000.0 for i in list(range(500,700,2))]
    colsample_bytree = [i/1000.0 for i in list(range(250,350,1))]
    min_child_weight = [i/1000.0 for i in list(range(250,550,3))]
    random.shuffle(rank_feature_num)
    random.shuffle(random_seed)
    random.shuffle(feature_num)
    random.shuffle(discret_feature_num)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    
    with open('params.pkl','wb') as f:
        cPickle.dump((random_seed,feature_num,rank_feature_num,discret_feature_num,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight),f)
    
    
    for i in list(range(36)):
        print ("iter:",i)
        pipeline(i,random_seed[i],feature_num[i],rank_feature_num[i],discret_feature_num[i],gamma[i],max_depth[i%3],lambd[i],subsample[i],colsample_bytree[i],min_child_weight[i])

    ##################################
    #  take average of xgb models    #
    ##################################


files = os.listdir('./preds')
pred = pd.read_csv('./preds/'+files[0])
uid = pred.uid
score = pred.score
for f in files[1:]:
    pred = pd.read_csv('./preds/'+f)
    score += pred.score

score /= len(files)

pred = pd.DataFrame(uid,columns=['uid'])
pred['score'] = score
pred.to_csv('avg_preds.csv',index=None,encoding='utf-8')


####################################################
#                                                  #
#                   xgb bagging                    #
#                                                  #
#################################################### 

###########################
#      prepare data       #
###########################

import pandas as pd
import xgboost as xgb
import sys
import random
import _pickle as cPickle
import os


train_nd = pd.read_csv('train_x_nd.csv')[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']]

train_dnull = pd.read_csv('train_x_null.csv')[['uid','discret_null']]

eleven_feature = ['n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','discret_null']
train_eleven = pd.merge(train_nd,train_dnull,on='uid')



discret_feature_score = pd.read_csv('discret_feature_score.csv')
fs = list(discret_feature_score.feature[0:500])#select important features only

discret_train = pd.read_csv("train_x_discretization.csv")[['uid']+fs]

#ranking 
rank_feature_score = pd.read_csv('rank_feature_score.csv')
fs = list(rank_feature_score.feature[0:500])
rank_train_x = pd.read_csv("train_x_rank.csv")
rank_train = rank_train_x[fs] / float(len(rank_train_x))
rank_train['uid'] = rank_train_x.uid


#raw
raw_feature_score = pd.read_csv('raw_feature_score.csv')
fs = list(raw_feature_score.feature[0:500])
raw_train_x = pd.read_csv("train_x.csv")[['uid']+fs]
raw_train_y = pd.read_csv("train_y.csv")

raw_train = pd.merge(raw_train_x,raw_train_y,on='uid')

#merge all except for raw

train = pd.merge(rank_train,discret_train,on='uid')
train = pd.merge(train,train_eleven,on='uid')
train.to_csv('final.csv',index=None)
final=pd.read_csv('final.csv')
