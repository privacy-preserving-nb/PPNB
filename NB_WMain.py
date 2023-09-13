import shutil
from tabnanny import check
import NB_WModule
import pandas as pd
import os
import time
import re
import numpy as np
import json
import natsort
import math
from sklearn.naive_bayes import CategoricalNB
from pathlib import Path
import heaan_sdk as heaan
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"  # set the number of CPU threads to use for parallel regions
# set key_dir_path
key_file_path = Path('./keys')

# set parameter
params = heaan.HEParameter.from_preset("FGb")

# init context and load all keys
context = heaan.Context(
    params,
    key_dir_path=key_file_path,
    load_keys="all",
    generate_keys=False,
)
num_slot = context.num_slots
log_num_slot = context.log_slots


def check_result(ctxt_path, y_class_num,datapath,real_):
    file_list2 = os.listdir(ctxt_path)
    file_list2 = natsort.natsorted(file_list2)
  
    return_result = []
    for ii,f in enumerate(file_list2): 
        if f == 'w':
            continue
        else:
            result_path = ctxt_path+f+'/'
            results = NB_WModule.decrypt_result(result_path,y_class_num,key_file_path)
            return_result.append(results)
    accuracy_sk_he = cal_accuracy_sk_he(return_result,datapath)
    accuracy_sk_real = cal_accuracy_sk_real(real_,datapath)
    accuracy_he_real = cal_accuracy_he_real(return_result,real_)
    return accuracy_sk_he, accuracy_sk_real, accuracy_he_real

def cal_accuracy_he_real(he_result,real_):
    count = 0
    total = len(he_result)
    for i in range(0,total):
        if he_result[i]==real_[i]:
            count+=1    
    return count/total

def cal_accuracy_sk_real(real_,datapath):
    tr = pd.read_csv(datapath+"train.csv")
    column = tr.columns
    X = tr[[i for i in column[:-1]]]
    Y = tr['label']

    catNB = CategoricalNB(alpha=0.01)
    catNB.fit(X, Y)

    te = pd.read_csv(datapath+"test.csv")
    test_X = te[[j for j in column[:-1]]]

    answer = list(catNB.predict(test_X))
    print("answer : ", answer)
    print("real : ",real_)

    count = 0
    total = len(answer)
    for i in range(0,total):
        if real_[i]==answer[i]:
            count+=1    
    return count/total

def cal_accuracy_sk_he(he_result,datapath):
    tr = pd.read_csv(datapath+"train.csv")
    column = tr.columns
    X = tr[[i for i in column[:-1]]]
    Y = tr['label']

    catNB = CategoricalNB(alpha=0.01)
    catNB.fit(X, Y)

    te = pd.read_csv(datapath+"test.csv")
    test_X = te[[j for j in column[:-1]]]

    answer = list(catNB.predict(test_X))
    print("he_result:",he_result)

    count = 0
    total = len(answer)
    for i in range(0,total):
        if he_result[i]==answer[i]:
            count+=1    
    return count/total


def car_encrypt():
    cell_col_max_list_car = '4,4,5,5,3,3,4'
    csv_data_path = './car_data/car_train.csv'
    data_ctxt_path = './car_ctxt/w/'
    result = {}
    path100 = './car_train_log/'

    test_data_path = "./car_data/test/"
    file_list = [re.sub('.csv','',i) for i in os.listdir(test_data_path)]
    file_list = natsort.natsorted(file_list)

    testdata_time = []
    for j,k in enumerate(file_list):
        test_data = test_data_path+k +'.csv'
        test_ctxt_path = './car_ctxt/test/'+str(j)+'/'
        ea = time.time()
        NB_WModule.inference_encrypt(test_data,test_ctxt_path,cell_col_max_list_car,context)
        eb = time.time()
        print("testdata encrypt time per 1: ",eb-ea)
        testdata_time.append(eb-ea)
    result["testdata time avg"] = np.mean(testdata_time)
    result["testdata time var"] = np.var(testdata_time)
    result["testdata time std"] = np.std(testdata_time)
    with open(path100+'test_encrypt.json', 'w') as f :
                f.write(json.dumps(result, sort_keys=True, indent=4, separators=(',', ': ')))

def car_check(n):
        cell_col_max_list_car = '4,4,5,5,3,3,4'
        learning_time_ = []
        csv_data_path = './car_data/car_train.csv'
        test_data_path = "./car_data/test/"
        data_ctxt_path = './car_ctxt/w/'
        model_ctxt_path = './car_train/w/'
        y_class_num=4
        datapath = './car_data/car_'
        path100 = './car_train_log/'
        file_list = [re.sub('.csv','',i) for i in os.listdir(test_data_path)]
        file_list = natsort.natsorted(file_list)

        alpha = 0.01
        real_ = [1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 3, 1, 4, 1, 1, 2, 2, 3, 1, 2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 4, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 3, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 3, 2, 1, 1, 2, 4, 2, 1, 2, 4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 4, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 4, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 2, 1, 1, 1, 1, 1, 4, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 4, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 4, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 4, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 1, 1, 4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 2, 2, 4, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 4, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 2, 1, 3, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 4, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 3, 2, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 4, 1, 1, 1, 3, 1, 1, 4, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 3, 1, 2, 1, 1, 1, 2, 1, 1, 3, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 4, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 3, 2, 1, 1, 4, 2, 1, 2, 3, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 3, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 4, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 3, 2, 1, 2, 2, 2, 3, 1, 2, 2, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 4, 2, 2, 1, 1, 2, 1, 1, 1, 3, 1, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 3, 1, 2, 1, 1, 4, 1, 4, 2, 4, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 3, 2, 2, 1, 4, 1, 1, 1, 4, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 1, 1, 3, 1, 1, 1, 2, 2, 4, 4, 1, 1, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 4, 1, 2, 1, 1, 1, 3, 1, 1, 2, 1, 2, 1, 3, 2, 2, 2, 1, 1, 2, 1, 1, 3, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 2, 2, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 2, 2, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 4, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 4, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 2, 4, 1, 2, 2, 1, 1, 1, 4, 1, 1, 1, 1, 1, 2, 1, 2, 1, 4, 1, 1, 1, 1, 3, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2, 1, 3, 1, 1, 1, 2, 1, 2, 1, 3, 1, 2, 3, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 4, 1, 1, 1, 1, 1, 2, 4, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 4, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 3, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 3, 2, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 4, 1, 1, 1, 1, 4, 1, 1, 4, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 4, 2, 1, 4, 2, 3, 1, 1, 1, 1, 2, 2, 1, 4, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1, 1, 2, 1, 2, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 4, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, 2, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3, 2, 1, 2, 2, 3, 1, 1, 1, 1, 2, 1, 4, 1, 1, 4, 2, 3, 1, 1, 1, 4, 1, 1, 1, 1, 1, 2, 3, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 3, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 2, 4, 1, 2, 1, 2, 4, 2, 1, 2, 1, 2, 1, 4, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 4, 2]
        result = {}
 
        print("##PROVIDER")
        pa = time.time()
        NB_WModule.data_encrypt(csv_data_path,data_ctxt_path,cell_col_max_list_car,context)
        pb = time.time()
        print("data encrypt time : ", pb-pa)
        infer_result = {}

        print("##LEARNING")
        la = time.time()
        NB_WModule.nb_learn(data_ctxt_path,cell_col_max_list_car,alpha,context, model_ctxt_path)
        lb = time.time()
        print("learning total time : ", lb-la)
        learning_time_.append(lb-la)
  
        inference_time=[]
        for j,k in enumerate(file_list):
            test_ctxt_path = './car_ctxt/test/'+str(j)+'/'
            ra = time.time()
            NB_WModule.nb_predict(test_ctxt_path,model_ctxt_path,y_class_num,cell_col_max_list_car,key_file_path)
            rb = time.time()
            print("predict time ",j,' data', rb-ra)
            inference_time.append(rb-ra)
        inference_time = inference_time[1:]
        infer_result["Inference time"] = inference_time
        infer_result["Inference time avg"] = np.mean(inference_time)
        infer_result["Inference time var"] = np.var(inference_time)
        infer_result["Inference time std"] = np.std(inference_time)

        accuracy_sk_he, accuracy_sk_real, accuracy_he_real = check_result('./car_ctxt/test/',y_class_num,datapath,real_)
        infer_result["he-sk Accuracy"] = accuracy_sk_he
        infer_result["real-sk Accuracy"] = accuracy_sk_real
        infer_result["real-he Accuracy"] = accuracy_he_real

        with open(path100+str(n)+'Inference0913.json', 'w') as f :
            f.write(json.dumps(infer_result, sort_keys=True, indent=4, separators=(',', ': ')))
        shutil.rmtree(model_ctxt_path)
        shutil.rmtree(data_ctxt_path)
    

def cancer_encrypt():
    cell_col_max_list_cancer = '10,10,10,10,10,10,10,10,10,2'
    csv_data_path = './cancer_data/cancer_train.csv'
    data_ctxt_path = './cancer_ctxt/w/'
    result = {}
    path100 = './cancer_train_log/'

    test_data_path = "./cancer_data/test/"
    file_list = [re.sub('.csv','',i) for i in os.listdir(test_data_path)]
    file_list = natsort.natsorted(file_list)

    testdata_time = []
    for j,k in enumerate(file_list):
        test_data = test_data_path+k +'.csv'
        test_ctxt_path = './cancer_ctxt/test/'+str(j)+'/'
        ea = time.time()
        NB_WModule.inference_encrypt(test_data,test_ctxt_path,cell_col_max_list_cancer,context)
        eb = time.time()
        print("testdata encrypt time per 1: ",eb-ea)
        testdata_time.append(eb-ea)
    result["testdata time avg"] = np.mean(testdata_time)
    result["testdata time var"] = np.var(testdata_time)
    result["testdata time std"] = np.std(testdata_time)
    with open(path100+'test_encrypt.json', 'w') as f :
                f.write(json.dumps(result, sort_keys=True, indent=4, separators=(',', ': ')))

def cancer_check(n):
        cell_col_max_list_cancer = '10,10,10,10,10,10,10,10,10,2'
        learning_time_ = []
        test_data_path = "./cancer_data/test/"
        csv_data_path = './cancer_data/cancer_train.csv'
        data_ctxt_path = './cancer_ctxt/w/'
        model_ctxt_path = './cancer_train/w/'
        datapath = './cancer_data/cancer_'
        y_class_num=2
        path100 = './cancer_train_log/'
        file_list = [re.sub('.csv','',i) for i in os.listdir(test_data_path)]
        file_list = natsort.natsorted(file_list)

        alpha = 0.01

        real_=[2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1]
        result = {}
   
        print("##PROVIDER")
        pa = time.time()
        NB_WModule.data_encrypt(csv_data_path,data_ctxt_path,cell_col_max_list_cancer,context)
        pb = time.time()
        print("data encrypt time : ", pb-pa)
        result["train data encrypt time"] = pb-pa

        infer_result = {}

        print("##LEARNING")
        la = time.time()
        NB_WModule.nb_learn(data_ctxt_path, cell_col_max_list_cancer,alpha,context, model_ctxt_path)
        lb = time.time()
        print("learning total time : ", lb-la)

        print("##INFERENCE")
        inference_time=[]
        for j,k in enumerate(file_list):
            test_ctxt_path = './cancer_ctxt/test/'+str(j)+'/'
            ra = time.time()
            NB_WModule.nb_predict(test_ctxt_path,model_ctxt_path,y_class_num,cell_col_max_list_cancer,key_file_path)
            rb = time.time()
            print("predict time ",j,' data', rb-ra)
            inference_time.append(rb-ra)
        inference_time = inference_time[1:]
        infer_result["Inference time"] = inference_time
        infer_result["Inference time avg"] = np.mean(inference_time)
        infer_result["Inference time var"] = np.var(inference_time)
        infer_result["Inference time std"] = np.std(inference_time)

        accuracy_sk_he, accuracy_sk_real, accuracy_he_real = check_result('./cancer_ctxt/test/',y_class_num,datapath,real_)
        infer_result["he-sk Accuracy"] = accuracy_sk_he
        infer_result["real-sk Accuracy"] = accuracy_sk_real
        infer_result["real-he Accuracy"] = accuracy_he_real

        with open(path100+str(n)+'Inference.json', 'w') as f :
            f.write(json.dumps(infer_result, sort_keys=True, indent=4, separators=(',', ': ')))
        shutil.rmtree(model_ctxt_path)
        shutil.rmtree(data_ctxt_path)



def a1_encrypt():
    cell_col_max_list_a1 = '66,2,2,2,2,2,2'
    csv_data_path = './a1_data/a1_train.csv'
    data_ctxt_path = './a1_ctxt/w/'
    result = {}
    path100 = './a1_train_log/'
    test_data_path = "./a1_data/test/"
    file_list = [re.sub('.csv','',i) for i in os.listdir(test_data_path)]
    file_list = natsort.natsorted(file_list)

    testdata_time = []
    for j,k in enumerate(file_list):
        test_data = test_data_path+k +'.csv'
        test_ctxt_path = './a1_ctxt/test/'+str(j)+'/'
        ea = time.time()
        NB_WModule.inference_encrypt(test_data,test_ctxt_path,cell_col_max_list_a1,context)
        eb = time.time()
        print("testdata encrypt time per 1: ",eb-ea)
        testdata_time.append(eb-ea)
    result["testdata time avg"] = np.mean(testdata_time)
    result["testdata time var"] = np.var(testdata_time)
    result["testdata time std"] = np.std(testdata_time)
    with open(path100+'test_encrypt.json', 'w') as f :
                f.write(json.dumps(result, sort_keys=True, indent=4, separators=(',', ': ')))

def a1_check(n):
        
        cpu_start = time.time()
        print('cpu start time: ', cpu_start)
        cell_col_max_list_a1 = '66,2,2,2,2,2,2'
        learning_time_ = []
        test_data_path = "./a1_data/test/"
        data_ctxt_path = './a1_ctxt/w/'
        model_ctxt_path = './a1_train/w/'
        y_class_num=2
        datapath = './a1_data/a1_'
        path100 = './a1_train_log/'
        file_list = [re.sub('.csv','',i) for i in os.listdir(test_data_path)]
        file_list = natsort.natsorted(file_list)
        csv_data_path = './a1_data/a1_train.csv'
        alpha = 0.01

        real_=[2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1]
        result = {}
        
        # for nn in range(n,n+6):
        print("##PROVIDER")
        pa = time.time()
        NB_WModule.data_encrypt(csv_data_path,data_ctxt_path,cell_col_max_list_a1,context)
        pb = time.time()
        print("data encrypt time : ", pb-pa)
        result["train data encrypt time"] = pb-pa

        infer_result = {}

        print("##LEARNING")
        la = time.time()
        NB_WModule.nb_learn(data_ctxt_path, cell_col_max_list_a1,alpha,context, model_ctxt_path)
        lb = time.time()
        print("learning total time : ", lb-la)
        learning_time_.append(lb-la)

        print("##Inference")
        inference_time=[]
        for j,k in enumerate(file_list):
            test_ctxt_path = './a1_ctxt/test/'+str(j)+'/'
            ra = time.time()
            NB_WModule.nb_predict(test_ctxt_path,model_ctxt_path,y_class_num,cell_col_max_list_a1,key_file_path)
            rb = time.time()
            inference_time.append(rb-ra)
            print("predict time ",j,' data', rb-ra)

        inference_time = inference_time[1:]
        infer_result["Inference time"] = inference_time
        infer_result["Inference time avg"] = np.mean(inference_time)
        infer_result["Inference time var"] = np.var(inference_time)
        infer_result["Inference time std"] = np.std(inference_time)

        accuracy_sk_he, accuracy_sk_real, accuracy_he_real = check_result('./a1_ctxt/test/',y_class_num,datapath,real_)
        infer_result["he-sk Accuracy"] = accuracy_sk_he
        infer_result["real-sk Accuracy"] = accuracy_sk_real
        infer_result["real-he Accuracy"] = accuracy_he_real

        with open(path100+str(n)+'Inference.json', 'w') as f :
            f.write(json.dumps(infer_result, sort_keys=True, indent=4, separators=(',', ': ')))
        shutil.rmtree(model_ctxt_path)
        shutil.rmtree(data_ctxt_path)

            

       
def a2_encrypt():
    
    cell_col_max_list_a2 = '66,2,2,2,2,2,2'
    csv_data_path = './a2_data/a2_train.csv'
    data_ctxt_path = './a2_ctxt/w/'
    result = {}
    path100 = './a2_train_log/'


    test_data_path = "./a2_data/test/"
    file_list = [re.sub('.csv','',i) for i in os.listdir(test_data_path)]
    file_list = natsort.natsorted(file_list)

    testdata_time = []
    for j,k in enumerate(file_list):
        test_data = test_data_path+k +'.csv'
        test_ctxt_path = './a2_ctxt/test/'+str(j)+'/'
        ea = time.time()
        NB_WModule.inference_encrypt(test_data,test_ctxt_path,cell_col_max_list_a2,context)
        eb = time.time()
        print("testdata encrypt time per 1: ",eb-ea)
        testdata_time.append(eb-ea)

    result["testdata time avg"] = np.mean(testdata_time)
    result["testdata time var"] = np.var(testdata_time)
    result["testdata time std"] = np.std(testdata_time)
    with open(path100+'test_encrypts.json', 'w') as f :
        f.write(json.dumps(result, sort_keys=True, indent=4, separators=(',', ': ')))

def a2_check(n):
        cell_col_max_list_a2 = '66,2,2,2,2,2,2'
        learning_time_ = []
        csv_data_path = './a2_data/a2_train.csv'
        test_data_path = "./a2_data/test/"
        data_ctxt_path = './a2_ctxt/w/'
        model_ctxt_path = './a2_train/w/'
        y_class_num=2
        datapath = './a2_data/a2_'
        path100 = './a2_train_log/'
        file_list = [re.sub('.csv','',i) for i in os.listdir(test_data_path)]
        file_list = natsort.natsorted(file_list)

        alpha = 0.01
        real_=[2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 2]
        result = {}

        print("##PROVIDER")
        pa = time.time()
        NB_WModule.data_encrypt(csv_data_path,data_ctxt_path,cell_col_max_list_a2,context)
        pb = time.time()
        print("data encrypt time : ", pb-pa)
        result["train data encrypt time"] = pb-pa

        infer_result = {}

        print("##LEARNING")
        la = time.time()
        NB_WModule.nb_learn(data_ctxt_path, cell_col_max_list_a2,alpha,context, model_ctxt_path)
        lb = time.time()
        print("learning total time : ", lb-la)
        learning_time_.append(lb-la)
        
        print("##Inference")
        inference_time=[]
        for j,k in enumerate(file_list):

            test_ctxt_path = './a2_ctxt/test/'+str(j)+'/'
            ra = time.time()
            NB_WModule.nb_predict(test_ctxt_path,model_ctxt_path,y_class_num,cell_col_max_list_a2,key_file_path)
            rb = time.time()
            print("predict time ",j,' data', rb-ra)
            inference_time.append(rb-ra)

        inference_time = inference_time[1:]
        infer_result["Inference time"] = inference_time
        infer_result["Inference time avg"] = np.mean(inference_time)
        infer_result["Inference time var"] = np.var(inference_time)
        infer_result["Inference time std"] = np.std(inference_time)

        accuracy_sk_he, accuracy_sk_real, accuracy_he_real = check_result('./a2_ctxt/test/',y_class_num,datapath,real_)
        infer_result["he-sk Accuracy"] = accuracy_sk_he
        infer_result["real-sk Accuracy"] = accuracy_sk_real
        infer_result["real-he Accuracy"] = accuracy_he_real

        with open(path100+str(n)+'Inference.json', 'w') as f :
            f.write(json.dumps(infer_result, sort_keys=True, indent=4, separators=(',', ': ')))
        shutil.rmtree(model_ctxt_path)
        shutil.rmtree(data_ctxt_path)


        
    
