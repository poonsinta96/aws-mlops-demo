
import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.exceptions import DataConversionWarning
from sklearn.compose import make_column_transformer

import boto3
os.system('pip install sagemaker')
import sagemaker
from sagemaker import get_execution_role

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
    parser.add_argument('--random-split', type=int, default=0)
    args, _ = parser.parse_known_args()
    
    print('Received arguments {}'.format(args))

    input_data_path = os.path.join('/opt/ml/processing/input', 'rawdata.csv')
    
    print('Reading input data from {}'.format(input_data_path))
    df = pd.read_csv(input_data_path)
    df.sample(frac=1)
    
    COLS = df.columns
    #newcolorder = ['PAY_AMT1','BILL_AMT1'] + list(COLS[1:])[:11] + list(COLS[1:])[12:17] + list(COLS[1:])[18:]
    newcolorder = list(COLS[1:])
    rest_col = newcolorder[:11]
    bill_col = newcolorder[11:17]
    pay_col = newcolorder[17:]
    
    
    split_ratio = args.train_test_split_ratio
    random_state=args.random_split
    
    #print('DF', df)
    
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Label', axis=1), df['Label'], 
                                                        test_size=split_ratio,
                                                        random_state=random_state)
    
    #print('X_TRAIN',X_train)
    #NOTE:, random_state=random_state
    #split into X_train_pay and X_train_bill
    X_train_rest = X_train.iloc[:,:11]
    X_train_bill = X_train.iloc[:,11:17]
    X_train_pay = X_train.iloc[:,17:] 
    
    #print('TRAIN_BILL',X_train_bill.head())
    #print('TRAIN_PAY',X_train_pay.head())
    #split into X_test_pay and X_test_bill
    X_test_rest = X_test.iloc[:,:11]
    X_test_bill = X_test.iloc[:,11:17]
    X_test_pay = X_test.iloc[:,17:] 
    
    #define scaler  
    bill_scaler = MinMaxScaler()
    pay_scaler = StandardScaler()

    # execute fit_transform on train
    X_train_bill_scaled = bill_scaler.fit_transform(X_train_bill)
    X_train_pay_scaled = pay_scaler.fit_transform(X_train_pay)
    print('Train_rest',X_train_rest)
    print('TRAIN_BILL_scaled',X_train_bill_scaled)
    print('TRAIN_PAY_scaled',X_train_pay_scaled)
    
    #execute transform on test
    X_test_bill_scaled = bill_scaler.transform(X_test_bill)
    X_test_pay_scaled = pay_scaler.transform(X_test_pay)
    #print('TEST_BILL_scaled',X_test_bill_scaled)
    #print('TEST_PAY_scaled',X_test_pay_scaled)
    #print(type(X_train_rest))
    #print(type(X_train_bill_scaled))
    """
    #not working
    preprocess = make_column_transformer(
        (['PAY_AMT1'], StandardScaler()),
        (['BILL_AMT1'], MinMaxScaler()),
    remainder='passthrough')
    
    print('Running preprocessing and feature engineering transformations')
    print(newcolorder)

    train_features = pd.DataFrame(preprocess.fit_transform(X_train), columns = newcolorder)
    test_features = pd.DataFrame(preprocess.transform(X_test), columns = newcolorder)
    """
    #print('X_train_rest',X_train_rest)
    train_index = X_train_rest.index
    train_features = X_train_rest
    train_features = train_features.join(pd.DataFrame(X_train_bill_scaled,columns=bill_col, index=train_index))
    train_features = train_features.join(pd.DataFrame(X_train_pay_scaled, columns=pay_col, index=train_index))
    test_features = X_test_rest    
    test_index = X_test_rest.index
    test_features = test_features.join(pd.DataFrame(X_test_bill_scaled,columns=bill_col, index=test_index))
    test_features = test_features.join(pd.DataFrame(X_test_pay_scaled, columns=pay_col, index=test_index))
    
    
    print('train_features',train_features)
    #print('test_features',test_features)
    
    # concat to ensure Label column is the first column in dataframe
    print('y_train',y_train)
    train_full = pd.DataFrame(y_train.values, columns=['Label'],index=train_index).join(train_features)
    test_full = pd.DataFrame(y_test.values, columns=['Label'],index=test_index).join(test_features)
    
    print('TRAIN_FULL',train_full)
    print('TEST_FULL',test_full)
    print('Train data shape after preprocessing: {}'.format(train_features.shape))
    print('Test data shape after preprocessing: {}'.format(test_features.shape))
    
    train_features_headers_output_path = os.path.join('/opt/ml/processing/train_headers', 'train_data_with_headers.csv')
    
    train_features_output_path = os.path.join('/opt/ml/processing/train', 'train_data.csv')
    
    test_features_output_path = os.path.join('/opt/ml/processing/test', 'test_data.csv')
    
    print('Saving training features to {}'.format(train_features_output_path))
    train_full.to_csv(train_features_output_path, header=False, index=False)
    print("Complete")
    
    print("Save training data with headers to {}".format(train_features_headers_output_path))
    train_full.to_csv(train_features_headers_output_path, index=False)
                 
    print('Saving test features to {}'.format(test_features_output_path))
    test_full.to_csv(test_features_output_path, header=False, index=False)
    print("Complete")

    
    
    
    
    #uploading testdata to s3 for monitoring demonstration 
    



    os.environ['AWS_DEFAULT_REGION'] = 'ap-southeast-1'
    role = get_execution_role()
    sess = sagemaker.Session()
    region = boto3.session.Session().region_name
    print("Region = {}".format(region))
    sm = boto3.Session().client('sagemaker')
    
    rawbucket = 'sagemaker-ap-southeast-1-692165707308'
    prefix = 'sagemaker-modelmonitor' # use this prefix to store all files pertaining to this workshop.
    dataprefix = prefix + '/data'
    testdataprefix = prefix + '/test_data'
    
    
        
    sess.upload_data(test_features_output_path,bucket=rawbucket,key_prefix=dataprefix)