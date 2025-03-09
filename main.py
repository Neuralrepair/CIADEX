import xgboost as xgb
import pandas as pd
import numpy as np
import pickle as pkl
from xgboost import plot_importance
import matplotlib.pyplot as plt
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import f_regression
Tsize = 15
testN = 6
def get_statistics_df(dfa,num_samples=100,num_n=10):
    results_df = pd.DataFrame()
    T = dfa.name
    df = dfa.drop('target', axis=1,errors='ignore')
    for i in range(num_samples):
        sampled_df = df.sample(n=num_n)
        mean_values = sampled_df.mean()
        mean_values.index=[i+"_mean" for i in mean_values.index]
        median_values = sampled_df.median()
        median_values.index=[i+"_median" for i in median_values.index]
        std_dev_values = sampled_df.std()
        std_dev_values.index=[i+"_std_dev" for i in std_dev_values.index]
        #skewness_values = sampled_df.apply(skew)
        #skewness_values.index=[i+"_skewness" for i in skewness_values.index]
        #kurtosis_values = sampled_df.apply(kurtosis)
        #kurtosis_values.index=[i+"_kurtosis" for i in kurtosis_values.index]
        temp_df = pd.concat(
            [
            mean_values,
            median_values,
            std_dev_values,
            #skewness_values,
            #kurtosis_values
            ],
        )
        temp_df = temp_df.to_frame().T 
        temp_df.index=[i]
        results_df = pd.concat([results_df, temp_df], axis=0)
    results_df.reset_index(drop=True, inplace=True)
    results_df['target'] = T
    return results_df
with open(r'.\Library\mingw-w64\model\NZMLLLscalar.pkl','rb') as f:
    scaler = pkl.load(f)
    model = xgb.Booster()
    model.load_model(r'.\Library\mingw-w64\model\NZMLLLxgb.model')

def get_statistics_df_test(df, num_samples=100, num_n=10, k=0):
    results_df = pd.DataFrame()
    for i in df.index:
        df_excluding_k = df.drop(k)
        sampled_df = df_excluding_k.sample(n=num_n-1)
        sampled_df = pd.concat([df.loc[[k]], sampled_df])
        mean_values = sampled_df.mean()
        mean_values.index = [i+"_mean" for i in mean_values.index]
        median_values = sampled_df.median()
        median_values.index = [i+"_median" for i in median_values.index]
        std_dev_values = sampled_df.std()
        std_dev_values.index = [i+"_std_dev" for i in std_dev_values.index]
        temp_df = pd.concat([mean_values, median_values, std_dev_values])
        temp_df = temp_df.to_frame().T 
        temp_df.index = [i]
        results_df = pd.concat([results_df, temp_df], axis=0)
    results_df.reset_index(drop=True, inplace=True)
    return results_df

def xgbtF(origin_df_b,model=model,scaler=scaler):
    result,probs=[],[]
    test_size=10
    for k in origin_df_b.index:
        combined_df_test1 = get_statistics_df_test(origin_df_b,num_samples=test_size,num_n=testN,k=k)
        if len(combined_df_test1.columns)==162:
            combined_df_test1.reset_index(inplace=True)
        else:
            combined_df_test1=combined_df_test1.reset_index(drop=True)
        #print(len(combined_df_test1.columns))
        X_test_k = scaler.transform(combined_df_test1.values)
        dtest = xgb.DMatrix(X_test_k)
        predictions = model.predict(dtest)
        r=np.quantile(predictions,0.7)+1.5;r=min(35,max(r,5))
        result.append(r)
        prob=np.sum(np.abs(predictions-r+2)<=3)/len(predictions)
        probs.append(prob)
    return result,probs
def train_model(root_dir,sr=0.5,tr=0.5):
    rs = glob.glob(root_dir+'/*.xlsx');print(rs)
    W = []
    groups = []
    trains,tests = [],[]
    for root_df in rs:
        if 'Control' in root_df:
            continue
        daici = int(os.path.basename(root_df)[5:7])
        WY = pd.read_excel(root_df,index_col = 'Unnamed: 0',sheet_name="WY")
        FG = pd.read_excel(root_df,index_col = 'Unnamed: 0',sheet_name="FG")
        FG.columns=[i +'_h' for i in FG.columns]
        WY=pd.concat([WY,FG],axis=1)
        if daici not in groups:
            groups.append(daici)
            WY['target'] = daici
        else:
            daici=daici+0.01
            WY['target'] = daici
            groups.append(daici)
        W.append(WY)
        trains.append(WY.iloc[8:28,:])
        tests.append(WY.iloc[40:,:])
    W=pd.concat(W);trains=pd.concat(trains);tests=pd.concat(tests)
    X=W.iloc[:,:-1]
    y=W.iloc[:,-1]
    X_train,y_train=trains.iloc[:,:-1],trains.iloc[:,-1]
    X_test,y_test=tests.iloc[:,:-1],tests.iloc[:,-1]
    
    #X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,shuffle=True)
    #sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    #for train_index, test_index in sss.split(X, y):
        #X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        #y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    df_train = pd.concat([X_train,y_train],axis=1)
    df_train.reset_index(drop=True,inplace=True)
    df_test = pd.concat([X_test,y_test],axis=1)
    df_test.reset_index(drop=True,inplace=True)
    combined_df_a = df_train.groupby('target').apply(get_statistics_df)
    combined_df_a.reset_index(drop=True,inplace=True)
    combined_df = combined_df_a.reset_index()
    #X = combined_df.drop('target', axis=1)
    X = scaler.transform(combined_df.values[:,:-1])
    y = combined_df_a['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tr, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'max_depth': 3,
        'eta': 0.1,
        'subsample':0.1,
        'alpha':5,
        'objective': 'reg:squarederror',
        #'objective':'reg:tweedie',
        #'objective':'reg:gamma',
        'learning':'',
        'eval_metric': 'mae'
    }
    epochs = 400
    global model
    model = xgb.train(params, dtrain, num_boost_round=30,xgb_model=model,learning_rate =0.01)
    #model.feature_names=list(combined_df.columns[:-1])
    #plot_importance(model, ax=ax)
    #print(list(combined_df.columns))
    #plt.savefig('a.png')

    # with open('NZMLLLscalar.pkl','wb') as f:
    #     pkl.dump(scaler,f)
    predictions = model.predict(dtest)
    #print('Accuracy:' ,np.sum(np.abs(predictions-y_test.values)<=2)/len(y_test))
    df_train.reset_index(drop=True,inplace=True)
    df_test.reset_index(drop=True,inplace=True)
    def getgroup(df):
        grouped =df.groupby('target')
        results_pred,results_prob,original_indices=[],[],[]
        for name,group in grouped:
            original_indices.extend(group.index)
            pred,prob=xgbtF(group.iloc[:,:-1],model,scaler)
            results_pred.append(pd.Series(pred,index=group.index))
            results_prob.append(pd.Series(prob,index=group.index))
        train_pred = pd.concat(results_pred).loc[original_indices]
        train_prob = pd.concat(results_prob).loc[original_indices]
        return train_pred,train_prob
    #train_pred, train_prob = xgbtF(df_train.iloc[:,:-1],model,scaler)
    train_pred,train_prob = getgroup(df_train)
    df_train['pred'] = train_pred
    df_train['prob'] = train_prob
    df_train['target'] = df_train['target'].astype('int')
    df_train.to_excel('train_data.xlsx')
    #test_pred, test_prob = xgbtF(df_test.iloc[:,:-1],model,scaler)
    
    test_pred,test_prob = getgroup(df_test)

    err = (np.array(test_pred)-df_test.values[:,-1])
    df_test['pred'] = test_pred
    df_test['prob'] = test_prob
    df_test['target'] = df_test['target'].astype('int')
    df_test.to_excel('valid_data.xlsx')
    return np.sum(np.abs(err)<=2)/len(err)

def xgbresultNZMLLLF(filename,model,scaler):
    df1 = pd.read_excel(filename,index_col='Unnamed: 0',sheet_name="WY")
    df2 = pd.read_excel(filename,index_col='Unnamed: 0',sheet_name="FG")
    df2.columns = [i+"_h" for i in df2.columns]
    origin_df_b = pd.concat([df1,df2],axis=1)
    origin_df_b.reset_index(drop=True,inplace=True)
    result,probs=[],[]

    test_size=100
    for k in range(len(origin_df_b)):
        combined_df_test1 = get_statistics_df_test(origin_df_b,num_samples=test_size,num_n=testN,k=k)
        if len(combined_df_test1.columns)==162:
            combined_df_test1.reset_index(inplace=True)
        else:
            combined_df_test1=combined_df_test1.reset_index(drop=True)
        X_test_k = scaler.transform(combined_df_test1.values)
        dtest = xgb.DMatrix(X_test_k)
        predictions = model.predict(dtest)
        r=1.5+np.quantile(predictions,0.7);r=max(min(r,35),5)
        result.append(r)
        prob=np.sum(np.abs(predictions+2-r)<=3)/len(predictions)
        probs.append(prob)
    return result,probs

def xgbresultNZMLLL(filename):
    df1 = pd.read_excel(filename,index_col='Unnamed: 0',sheet_name="WY")
    df2 = pd.read_excel(filename,index_col='Unnamed: 0',sheet_name="FG")
    df2.columns = [i+"_h" for i in df2.columns]
    origin_df_b = pd.concat([df1,df2],axis=1)
    origin_df_b.reset_index(drop=True,inplace=True)
    #result_df=get_statistics_df(df3,num_samples=100,num_n=testN)
    #X_NZM = result_df.reset_index(inplace=False)
    with open(r'.\Library\mingw-w64\model\NZMLLLscalar.pkl','rb') as f:
        scaler = pkl.load(f)
    ##print(result_df)
    model = xgb.Booster()
    model.load_model(r'.\Library\mingw-w64\model\NZMLLLxgb.model')
    result,probs=[],[]

    test_size=100
    for k in range(len(origin_df_b)):
        combined_df_test1 = get_statistics_df_test(origin_df_b,num_samples=test_size,num_n=testN,k=k)
        combined_df_test1=combined_df_test1.reset_index()
        X_test_k = scaler.transform(combined_df_test1.values)
        dtest = xgb.DMatrix(X_test_k)
        predictions = model.predict(dtest)
        r=np.median(predictions)+1
        result.append(r)
        prob=np.sum(2*np.abs(predictions-r)<=1)/test_size
        probs.append(prob)
    return result,probs


 

    


if __name__ == "__main__":
    filename = input("Please enter filename:") 
    R=[]
    for i in range(1,10):
        result= train_model(filename,sr=0.6,tr=0.1*i)
        R.append(result)
    print(R)

    
