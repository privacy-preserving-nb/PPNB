from cgi import print_environ
from cmath import log
import heaan_sdk as heaan
import os
import time
import numpy as np
import pandas as pd
import math
import json
import re
import natsort
import NB_log
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "16"  # set the number of CPU threads to use for parallel regions


key_dir_path = Path('./keys')

# set parameter
params = heaan.HEParameter.from_preset("FGb")

# init context and load all keys
context = heaan.Context(
    params,
    key_dir_path=key_dir_path,
    load_keys="all",
    generate_keys=False,
)
num_slot = context.num_slots
log_num_slot = context.log_slots    

#####################################################################################
###############           Naive Bayesian classifier LEARNING          ###############
#####################################################################################
def data_encrypt(csv_file_path, ctxt_path, cell_col_max_list,context):
    try:
        os.makedirs(name=ctxt_path, mode=0o775, exist_ok=True)
    except Exception as e:
        print("[Error] Could not make train table directory: ", e)
        return
   
    tmp = pd.read_csv(csv_file_path)
    col_name = list(tmp.columns)
    col_max_list = [int(str(cell_col_max).strip()) for cell_col_max in str(cell_col_max_list).split(',')]

    for j, c in enumerate(col_name):
        if 'label' in c:
            m = col_max_list[-1]
        else:
            m = col_max_list[int(c[-1])]
        for n in range(1,m+1):
            _add_one_zero_column(tmp,c,n)

    data = tmp.drop(columns=col_name, axis=1)
    
    for cname in data.columns:
  
        msg = heaan.Block(context,encrypted = False)
        for index in range(data[cname].size):
            msg[index]=data[cname][index]
        if ("Unnamed" not in cname):
            c = msg.encrypt(inplace=False)
            c.save(ctxt_path+cname+".ctxt")
           

    json_opend={}
    json_opend["num_rows"] = len(data)
    json_opend["bin_col_names"] = list(data.columns)
    json_opend["bin_num_lists"] = []

    for i , c in enumerate(col_name):
        if 'label' not in c:
            json_opend["bin_num_lists"].append(c + ":" + ','.join([str(j) for j in range(1,col_max_list[int(c[-1])]+1)]))
        else: 
            json_opend["bin_num_lists"].append(c + ":" + ','.join([str(j) for j in range(1,col_max_list[-1]+1)]))

    with open(ctxt_path+"metadata.json", 'w') as f:
        f.write(json.dumps(json_opend, sort_keys=True, indent=4, separators=(',', ': ')))

def nb_learn(data_ctxt_path,cell_col_max_list,alpha,context, model_ctxt_path):

    try:
        os.makedirs(name=model_ctxt_path, mode=0o775, exist_ok=True)
    except Exception as e:
        print("[Error] Could not make train table directory: ", e)
        return
    

    col_max_list = [int(str(cell_col_max).strip()) for cell_col_max in str(cell_col_max_list).split(',')]

    with open(data_ctxt_path+"metadata.json",'r') as file:
            data_json_opend = json.load(file)
    data_cell_name_list = data_json_opend['bin_col_names']
    total_num = data_json_opend['num_rows']
    total_num_list = [total_num]*num_slot

    total_num_msg = heaan.Block(context,encrypted = False, data = total_num_list)
    total_num_ctxt = total_num_msg.encrypt()
    total_num_ctxt.save(model_ctxt_path+'total_num.ctxt')

    data_cdict = load_ctxt(data_cell_name_list,data_ctxt_path,context)
    x_cols = []
    y_cols = []
    for i in data_cell_name_list:
        if 'X' in i:
            x_cols.append(i)
        else:
            y_cols.append(i)

    y_labels = col_max_list[-1]
    for y_label in range(1,y_labels+1):
        for i,p in enumerate(x_cols):
       
            cname = str(y_label)+ "count"+p+"_"
        
            out_ctxt = data_cdict[str(y_label)+'Y_'] * data_cdict[p]

            check_boot(out_ctxt)
            
            
            rot_ctxt = rotate_sum(out_ctxt)
            check_boot(rot_ctxt)

            rot_ctxt = rot_ctxt + alpha
            rot_ctxt.save(model_ctxt_path+cname+".ctxt")

        yname = str(y_label)+'acountY_'
        print(yname)
        rot_ctxt2 = rotate_sum(data_cdict[str(y_label)+'Y_'])
        check_boot(rot_ctxt2)
        rot_ctxt2.save(model_ctxt_path+yname+".ctxt")

        
    fl = [re.sub('.ctxt','',i) for i in os.listdir(model_ctxt_path)]
    fl = natsort.natsorted(fl)

    data_cdict = load_ctxt(fl,model_ctxt_path,context)
    
    
    inverse_ctxt, inverse_index = make_inverse_ctxt(model_ctxt_path, context,alpha,col_max_list)


    
    rotate_ctxt = make_rotate_ctxt(inverse_ctxt,inverse_index,context)


    m_25_ctxt = make_25_ctxt(rotate_ctxt)

    
    msg1 = [0]*num_slot
    for i in range(0,len(inverse_index)*25):
        msg1[i] = 1
    
    he_msg1 = heaan.Block(context,encrypted=False, data=msg1)

    m_25_ctxt = m_25_ctxt*he_msg1
    check_boot(m_25_ctxt)

    
    find_log_ctxt = make_log_ctxt(m_25_ctxt,inverse_index,model_ctxt_path, context)

   
    unit_size = 0
    for i in range(len(col_max_list)-1):
        unit_size+=int(col_max_list[i])
        
    unit_size+=1
    rs = [0]*32768
    st=0
    for _ in range(unit_size*col_max_list[-1]):
        rs[st]=1
        st +=25
 
    rs_msg = heaan.Block(context,encrypted=False,data=rs)
    
    final_result = find_log_ctxt * rs_msg

    final_result = make_25_ctxt(final_result)
    
    final_result.save(model_ctxt_path+"model.ctxt")

    flist = os.listdir(model_ctxt_path)
    for i in flist:
        if 'model' not in i:
            os.remove(model_ctxt_path+i)

#####################################################################################
###########################           INFERENCE          ############################
#####################################################################################
def inference_encrypt(csv_file_path, ctxt_path, cell_col_max_list,context):
    csv_file_path = str(csv_file_path)
    ctxt_path = str(ctxt_path)
    cell_col_max_list = str(cell_col_max_list)
    col_max_list = [int(str(cell_col_max).strip()) for cell_col_max in str(cell_col_max_list).split(',')]
    num_class = col_max_list[-1]
    try:
        os.makedirs(name=ctxt_path, mode=0o775, exist_ok=True)
    except Exception as e:
        print("[Error] Could not make train table directory: ", e)
        return
 
    input_ = pd.read_csv(csv_file_path) 
 
    unit_size = 0
    for i in range(len(col_max_list)-1):
        unit_size+=col_max_list[i]

    unit_size+=1
    total_size = unit_size * num_class*25

    ndata = num_slot//total_size
    required_ctxt = len(input_)//ndata if (len(input_) % ndata ==0) else len(input_)//ndata+1 

    input_li = [[0]*num_slot for _ in range(required_ctxt)]
    cur_ctxt=0
    cur_idx=0
    cur_data=0
    for i in range(len(input_)):
        cur_item = input_.iloc[i].to_list()
        for k in range(num_class):
            input_li[int(cur_ctxt)][int(cur_idx+k*unit_size)*25]=1 ## Add y probability
        for j in range(len(col_max_list)-1): ## y is excluded
            for k in range(num_class):
                input_li[int(cur_ctxt)][int(cur_idx+cur_item[j]+k*unit_size)*25]=1
            cur_idx+=col_max_list[j]
        cur_idx+=1 ## for y
        cur_data+=1
        if (cur_data == ndata):
            cur_data=0
            cur_ctxt+=1
            cur_idx=0


    for i in range(required_ctxt):
      
        msg = heaan.Block(context,encrypted = False, data = input_li[i])
        ctxt_out = msg.encrypt(inplace=False)
        ctxt_out.save(ctxt_path+"/input_"+str(i)+"_NB.ctxt")

def nb_predict(test_ctxt_path,model_ctxt_path, ycn, cell_col_max_list,key_file_path):

    y_class_num = int(ycn)
    cell_col_max_list = str(cell_col_max_list)
    col_max_list = [int(str(cell_col_max).strip()) for cell_col_max in str(cell_col_max_list).split(',')]
    unit_size=0
    for i in range(len(col_max_list)-1):
        unit_size+=col_max_list[i]
    unit_size+=1

    empty_msg= heaan.Block(context,encrypted = False)
    model_ctxt = empty_msg.encrypt(inplace=False) 
    model_ctxt = model_ctxt.load(model_ctxt_path+"model.ctxt")
    
    feature_length = len(col_max_list)-1

    log_ = [-1/(np.log(1/32768)*(feature_length+1))]*num_slot
  
    logs = heaan.Block(context,data = log_)
    logs.encrypt()
    

    check_boot(model_ctxt)
    
    empty_msg= heaan.Block(context,encrypted = False)
    input_ctxt = empty_msg.encrypt(inplace=False) 
    input_ctxt = input_ctxt.load(test_ctxt_path+"input_0_NB.ctxt")
    check_boot(input_ctxt)

    input_ctxt = input_ctxt * model_ctxt
    check_boot(input_ctxt)
    
    input_ctxt = input_ctxt * logs
    check_boot(input_ctxt)

    u_size = unit_size
    rot_size = u_size//2 if (u_size % 2 ==0) else u_size//2+1 

    masku = [0]*num_slot 
    maskd = [0]*num_slot 
    
    num_iter = num_slot//(u_size*25)
    for i in range(num_iter):
        for j in range(u_size//2):
            maskd[i*u_size*25+j*25]=1
        for j in range(rot_size):
            masku[i*u_size*25+j*25]=1

    
    tmp_ctxt = input_ctxt.__lshift__(rot_size*25)
    
    check_boot(tmp_ctxt)


    masku_msg = heaan.Block(context,data = masku)

   
    input_ctxt = input_ctxt * masku_msg
    check_boot(input_ctxt)

    maskd_msg = heaan.Block(context, data = maskd)
    
    tmp_ctxt = tmp_ctxt * maskd_msg
    check_boot(tmp_ctxt)

    input_ctxt = input_ctxt + tmp_ctxt

    
    while(rot_size>1):
        rot_size = rot_size//2 if (rot_size % 2 ==0) else rot_size//2+1     
       
        tmp_ctxt = input_ctxt.__lshift__(rot_size*25)
        check_boot(tmp_ctxt)
      
        input_ctxt = input_ctxt + tmp_ctxt

    input_ctxt_duplicate = input_ctxt


    tmp_msg = [0]*num_slot

    for i in range(y_class_num):
        tmp_msg[i]=1

        tmp_ = heaan.Block(context,encrypted = False, data = [0]*num_slot) ## 이렇게하면 block은 ciphertext가 아니라 message임
        tmp_.encrypt()

        msg=[0]*num_slot
        msg[i]=1
      
        one_msg = heaan.Block(context, data = msg)
        
        if i ==0:
         
            result_ctxt = input_ctxt * one_msg
            check_boot(result_ctxt)
        else : 
     
            input_ctxt_duplicate = input_ctxt_duplicate.__lshift__(25*(unit_size)-1)
            
            check_boot(input_ctxt_duplicate)

            tmp_ = input_ctxt_duplicate * one_msg
            check_boot(tmp_)

            result_ctxt = tmp_ + result_ctxt

    tmp_ctxt2 = heaan.Block(context, encrypted = False, data = tmp_msg)
    tmp_ctxt2.encrypt()

    result_ctxt = result_ctxt * tmp_ctxt2
    check_boot(result_ctxt)
    
    if y_class_num != 2:
        final_result = findMaxPos(result_ctxt,context,log_num_slot,y_class_num)
    
    elif y_class_num ==2 : 
    
        result_duplicate = result_ctxt
        result_duplicate = result_duplicate.__lshift__(1)
        
        check_boot(result_duplicate)
   
        result_duplicate = result_duplicate * (-1)
        check_boot(result_duplicate)
      
        result_ctxt = result_ctxt + result_duplicate
     
        final_result = result_ctxt.sign(inplace = False, log_range=0)
        
    print("### predict end ###")
    final_result.save(test_ctxt_path+"result.ctxt")

def decrypt_result(model_ctxt_path,ycn,key_file_path):

    y_class_num = int(ycn)
    
    empty_msg= heaan.Block(context,encrypted = False)
    result_ctxt = empty_msg.encrypt(inplace=False) 

    result_ctxt = result_ctxt.load(model_ctxt_path+'result.ctxt')

    result_msg = result_ctxt.decrypt(inplace=False)

    if y_class_num==2:
        num = round(result_msg[0].real)
        if num ==1:
            return 1
        else : 
            return 2
    else : 
        num_list = []
        for i in range(y_class_num):
            num_list.append(round(result_msg[i].real))
        return (num_list.index(max(num_list)))+1


#####################################################################################
########################           INNER FUNCTIONS         ##########################
#####################################################################################
def make_rotate_ctxt(input_ctxt, index_dict,context):
 
    one_ = [0]*num_slot

    one_msg = heaan.Block(context,encrypted = False,data=one_)

    return_ctxt = one_msg.encrypt(inplace=False)

   
    duplicate_ctxt = input_ctxt

    for i in range(len(index_dict)):
    
        tmp_ctxt = one_msg
        mask_ = [0]*num_slot
        mask_[i*25]=1
  
        mask_msg= heaan.Block(context,encrypted = False,data=mask_)
        
        if i ==0:
       
            tmp_ctxt = input_ctxt * mask_msg
            check_boot(tmp_ctxt)
       
            return_ctxt = tmp_ctxt + return_ctxt
        else:

            duplicate_ctxt = duplicate_ctxt.__rshift__(24)
            
            check_boot(duplicate_ctxt)
        
            tmp_ctxt = duplicate_ctxt * mask_msg
            check_boot(tmp_ctxt)
            check_boot(duplicate_ctxt)
    
            return_ctxt = return_ctxt + tmp_ctxt

    return return_ctxt

def make_25_ctxt(ctxt):
    origin = ctxt
    tmp_ = ctxt
    tmp_ = ctxt.__rshift__(1) 
    check_boot(ctxt)
    ctxt = ctxt + tmp_
    
    tmp_ = ctxt.__rshift__(2)
    check_boot(ctxt)
    ctxt = ctxt + tmp_
    

    tmp_ = ctxt.__rshift__(4)   
    check_boot(ctxt)
    ctxt = ctxt + tmp_
    
    ctxt_8_ = ctxt
    ctxt_8_ = ctxt.__rshift__(16)
    check_boot(ctxt_8_)
    
    tmp_ = ctxt.__rshift__(8)
    check_boot(ctxt)
    ctxt = ctxt + tmp_
    ctxt = ctxt + ctxt_8_
  
    origin = origin.__rshift__(24)
    check_boot(origin)

    ctxt = ctxt + origin
    check_boot(ctxt)

    return ctxt

def make_inverse_ctxt(model_ctxt_path, context,alpha, col_max_list):

    f_list = os.listdir(model_ctxt_path)
    file_list = []
    for i in f_list:
        if 'total' not in i:
            file_list.append(re.sub('.ctxt','',i))
    file_list = natsort.natsorted(file_list)
    ta = time.time()
    model_cdict = load_ctxt(file_list,model_ctxt_path,context)
    
    one_ = [0]*num_slot
    inverse_index = {}

    one_msg = heaan.Block(context, encrypted = False, data = one_)
    inverse_ctxt = one_msg.encrypt(inplace=False)


    count_ctxt = one_msg

    for i,n in enumerate(file_list):        
        if 'log' not in n : 
            inverse_index[i] = n

    empty_msg= heaan.Block(context,encrypted = False)
    total_num_ctxt = empty_msg.encrypt(inplace=False)        
    total_num_ctxt = total_num_ctxt.load(model_ctxt_path+'total_num.ctxt')
 

    for i,n in enumerate(inverse_index):

        tmp = [0]*num_slot
        tmp[i]=1
   
        tmp_msg = heaan.Block(context,encrypted = False, data = tmp)

        if 'X' in inverse_index[n]:
            ## nominator

            model_cdict[inverse_index[n]] = model_cdict[inverse_index[n]] * tmp_msg
            check_boot(model_cdict[inverse_index[n]])

            count_ctxt = model_cdict[inverse_index[n]] + count_ctxt
            
        else:

            tmp_ctxt1 = one_msg
            ## make nominator
            
            tmp_ctxt1 = model_cdict[inverse_index[n]] * tmp_msg
            check_boot(tmp_ctxt1)
    
            count_ctxt = tmp_ctxt1 + count_ctxt

    ## make denominator

    tmp_ctxt=one_msg

    tc = [0]*num_slot

    unit_size = 0        
    for i in range(len(col_max_list)-1):
        unit_size+=col_max_list[i]
    unit_size+=1
    
    #total count in each Y slot
    y_loca = 0
    for _ in range(col_max_list[-1]):
        tc[y_loca]=1
        y_loca+=unit_size
    
    y_loc_msg = heaan.Block(context,encrypted = False, data = tc)
   
    total_num_ctxt = total_num_ctxt * y_loc_msg
    
    check_boot(total_num_ctxt)

   
    inverse_ctxt = total_num_ctxt + inverse_ctxt
   
    for i in range(1,col_max_list[-1]+1):
        tc = [0]*num_slot
        for j in range(unit_size*(i-1),unit_size*i):
            if j ==unit_size*(i-1):
                pass
            else : 
                tc[j]=1
 
        y_tmp_msg = heaan.Block(context,encrypted = False, data = tc)

        model_cdict[str(i)+'acountY_'] = model_cdict[str(i)+'acountY_']*y_tmp_msg
   
        inverse_ctxt = model_cdict[str(i)+'acountY_'] + inverse_ctxt

    nc = [0]*num_slot
    unit_ = [0]*unit_size

    cur=1
    for j in range(len(col_max_list[:-1])):
        ni = col_max_list[j]
        for i in range(cur,cur+ni):
            unit_[i]=alpha*ni
        cur+=ni

    unit_msg = unit_*col_max_list[-1]
    for i,ui in enumerate(unit_msg):
        nc[i]=ui

    nc_msg = heaan.Block(context,encrypted = False, data = nc)
    nc_ctxt = nc_msg.encrypt(inplace=False)

    inverse_ctxt = nc_ctxt+inverse_ctxt


    inverse_ctxt = inverse_ctxt*0.0001

    count_ctxt = count_ctxt*0.0001
    
    check_boot(inverse_ctxt)
    check_boot(count_ctxt)
 
 
    inverse_ctxt = inverse_ctxt.inverse(greater_than_one = False)
    check_boot(inverse_ctxt)

  
    inverse_ctxt = inverse_ctxt * count_ctxt

    return inverse_ctxt, inverse_index

def make_log_ctxt(m_25_ctxt,index_dict,model_ctxt_path, context):

    check_boot(m_25_ctxt)
    
    log_ctxt = NB_log.find_log(m_25_ctxt,log_num_slot,context)
   

    return log_ctxt


def findMax4(c, context, logN, ndata):
  
    if (ndata==1): 
        return c
    check_boot(c)
    
    if (ndata % 4 !=0):
        i=ndata
  
        msg = heaan.Block(context,encrypted = False, data = [0]*num_slot)
        while (i % 4 !=0):
            msg[i]=0.00000
            i+=1
        ndata=i
 
        c = c + msg
    
    ms1=[0]*num_slot
    for i in range(num_slot):
        ms1[i]=0
    for i in range(ndata//4):
        ms1[i]=1


    msg1 = heaan.Block(context,encrypted = False, data = ms1)
    msg1.encrypt()


    ca = c*msg1

    ctmp1 = c.__lshift__(ndata//4)     

    cb = ctmp1 * msg1
    
    
    ctmp1 = c.__lshift__(ndata//2)
   
    cc = ctmp1 * msg1
    
    ctmp1 = c.__lshift__(ndata*3//4)
  
    cd = ctmp1 * msg1

    check_boot(ca)
    check_boot(cb)
    check_boot(cc)
    check_boot(cd)


    c1= ca-cb

    c2 = cb - cc

    c3 = cc - cd

    c4 = cd - ca

    c5 = ca - cc

    c6 = cb-cd    

    ctmp1 = c2.__rshift__(ndata//4)

    ctmp1 = ctmp1 + c1

   
    ctmp2 = c3.__rshift__(ndata//2)
    
    ctmp1 = ctmp1 + ctmp2
    
  
    ctmp2 = c4.__rshift__(ndata*3//4)
 
    ctmp1 = ctmp1 + ctmp2
    
    ctmp2 = c5.__rshift__(ndata)
    
    ctmp1 = ctmp1 + ctmp2
    
    ctmp2 = c6.__rshift__(5*ndata//4)
 
    ctmp1 = ctmp1 + ctmp2
    
    c0 = ctmp1.sign(inplace = False, log_range=0) ## input ctxt range : -1 ~ 0 (log value)
    check_boot(c0)
    
 
    c0_c= c0
    
    mkmsg = [1.0]*num_slot
    
    mkall = heaan.Block(context,encrypted = False, data=mkmsg)

    c0 = c0 + mkall

    c0 = c0 * 0.5
    check_boot(c0)


    ceq = c0_c * c0_c
    check_boot(ceq)

    ceq = ceq.__neg__()

    ceq = ceq + mkall
    
    mk1 = msg1
    
    mk2 = mk1.__rshift__(ndata//4)
    
 
    mk3 = mk2.__rshift__(ndata//4)
    

    mk4 = mk3.__rshift__(ndata//4)
    

    mk5 = mk4.__rshift__(ndata//4)
    

    mk6 = mk5.__rshift__(ndata//4)

    c_neg = c0
    
    c_neg = c0.__neg__()


    c_neg = c_neg+mkall

    c0n = c0
    ctmp1 = c0n * mk1
    
    ctxt=c0n
 
    c_ab = ctmp1

    c0 = ctxt

    ctmp2 = c_neg * mk4    

    ctmp2 = ctmp2.__lshift__(ndata*3//4)
    
    ctmp1 = ctmp1 * ctmp2


    ctmp2 = c0 * mk5

    ctmp2 = ctmp2.__lshift__(ndata)

    cca = ctmp1 * ctmp2

    ctmp1 = c_neg * mk1

    ctmp2 = c0 * mk2

    ctmp2 = ctmp2.__lshift__(ndata//4)

    c_bc = ctmp2

    ctmp1 = ctmp1 * ctmp2

    ctmp2 = c0*mk6
  
    ctmp2 = ctmp2.__lshift__(ndata*5//4)
    
    ccb = ctmp1 * ctmp2

  
    ctmp1 = c_neg *mk2
  
    ctmp1 = ctmp1.__lshift__(ndata//4)
    

    ctmp2 = c0 * mk3
    
    ctmp2 = ctmp2.__lshift__(ndata//2)

    c_cd = ctmp2

    ctmp1 = ctmp1 * ctmp2
  
    ctmp2 = c_neg * mk5
 
    ctmp2 = ctmp2.__lshift__(ndata)
    

    ccc = ctmp1 * ctmp2
    
 
    ctmp1 = c_neg * mk3
  
    ctmp1 = ctmp1.__lshift__(ndata//2)

    ctmp2 = c0 * mk4
  
    ctmp2 = ctmp2.__lshift__(ndata*3//4)
    
    cda = ctmp2
    
    ctmp1 = ctmp1 * ctmp2


    ctmp2 = c_neg * mk6
  
    ctmp2 = ctmp2.__lshift__(ndata*5//4)

    ccd= ctmp1 * ctmp2

    check_boot(cca)
    check_boot(ccb)
    check_boot(ccc)
    check_boot(ccd)

  
    cca = cca * ca

    ccb = ccb * cb
 
    ccc = ccc * cc
   
    ccd = ccd * cd
    
   
    cout = cca
 
    cout = cout * ccb
 
    cout = cout * ccc
 
    cout = cout * ccd

    check_boot(cout)

   
    cneq = ceq.__neg__()
 
    cneq = cneq + mkall
   
    cneq_da = cneq.__lshift__(ndata*3//4)
   
    cneq_da = cneq_da * mk1

    cneq_bc = cneq.__lshift__(ndata//4)
    

    cneq_bc =cneq_bc*mk1

   
    ceq_ab = ceq * mk1

    ceq_bc = ceq.__lshift__(ndata//4)
   
    ceq_bc = ceq_bc * mk1

    
    ceq_cd = ceq.__lshift__(ndata//2)
  
    ceq_cd = ceq_cd * mk1

    ceq_da = cneq_da.__neg__()
   
    ceq_da = ceq_da + mk1
    check_boot(ceq)
    check_boot(ceq_ab)
    check_boot(ceq_bc)
    check_boot(ceq_cd)
    check_boot(ceq_da)
    
 
    ctmp2 = ceq_ab * ceq_bc
 
    ctmp1 = ctmp2 * c_cd
    ctmp1.bootstrap()
   
    c_cond3 = ctmp1

    ctmp1 = ceq_bc * ceq_cd
    ctmp1 = ctmp1 * cda
    c_cond3 = c_cond3 + ctmp1
    
    ctmp1 = ceq_cd * ceq_da
    ctmp1 = ctmp1 * c_ab
    c_cond3 - c_cond3 + ctmp1

    ctmp1 = ceq_ab + ceq_da
    ctmp1 = ctmp1 * c_bc
    c_cond3 = c_cond3 + ctmp1

    c_cond4 = ceq_cd* ctmp2

    c_cond3.bootstrap()
    c_cond4.bootstrap()

    c_tba = c_cond3 * 0.333333333
    c_tba.bootstrap()
    c_tba = c_tba + mkall
    c_cond4.bootstrap()
    ctmp1 = c_cond4 + mkall
    c_tba = c_tba * ctmp1
    cout = cout * c_tba
    cout.bootstrap()

    return findMax4(cout, context, logN, ndata//4)

def findMaxPos(c,context,logN,ndata):
    
    cmax = findMax4(c,context,logN,ndata)

    for i in range(logN-1):
    
        ctmp = cmax.__rshift__(pow(2,i))

        cmax = cmax + ctmp
 
    c = c - cmax
  
    c = c + 0.000001

    c_red = c.greater_than_zero()

    c_out=selectRandomOnePos(c_red,context, logN,ndata)
    return c_out

def selectRandomOnePos(c_red,context,logN,ndata):
 
    m0 = heaan.Block(context, data = [0.0]*num_slot)
    c_sel = m0.encrypt(inplace=False)
  
    rando = np.random.permutation(ndata)
    ctmp1 = c_red
  
 
    if (ctmp1.level-4< context.min_level_for_bootstrap):
         ctmp1.bootstrap()
 

    m0[0]=1.0
    
    for l in rando:
        if (l>0):
            
            if (ctmp1.level-4< context.min_level_for_bootstrap):
                ctmp1.bootstrap()

            if (c_sel.level-4< context.min_level_for_bootstrap):
                c_sel.bootstrap()

            ctmp1 = ctmp1.__lshift__(l)
           
            ctmp2 = ctmp1 * c_sel
            ctmp1 = ctmp1 - ctmp2
            ctmp2 = ctmp1 * m0
            ctmp1 = ctmp1.__rshift__(l)
            
            c_sel = c_sel + ctmp2
            
        else:

            if (ctmp1.level - 4< context.min_level_for_bootstrap):
                ctmp1.bootstrap()

            if (c_sel.level-4< context.min_level_for_bootstrap):
                c_sel.bootstrap()
           
            ctmp2 = c_sel * ctmp1
            ctmp1 = ctmp1 - ctmp2
            ctmp2 = ctmp1 * m0
            c_sel = c_sel + ctmp2

    if (ctmp1.level - 4 < context.min_level_for_bootstrap):
        ctmp1.bootstrap()

    return ctmp1

def _add_one_zero_column(tmp, column_name,n):
    result = [0]*len(tmp)
    for i,cc in enumerate(tmp[column_name]):
        if cc == n:
            result[i] = 1
        if "feature" in column_name : 
            tmp["X_"+column_name[-1]+"_"+str(n)]=result
        else :
            tmp[str(n)+"Y_"] = result        

def load_ctxt(fn_list,ctxt_path,context):
   
    out_cdict={}
    for cname in fn_list:
        empty_msg= heaan.Block(context,encrypted = True)
        ctxt = empty_msg.load(ctxt_path+cname+".ctxt")      
        out_cdict[cname]=ctxt
    return out_cdict

def rotate_sum(input_ctxt):
    for i in range(int(np.log2(num_slot))):
        tmp_ctxt = input_ctxt.__lshift__(2**i)
        check_boot(tmp_ctxt)
        input_ctxt = input_ctxt + tmp_ctxt
    return input_ctxt

def check_boot(x):
    if x.level==3:
        x.bootstrap()
    elif x.level<3:
        exit(0)
    return x

def print_ctxt(c,size):
    m = c.decrypt(inplace=False)
    for i in range(size):
        print(i,m[i])
        if (math.isnan(m[i].real)):
            print ("nan detected..stop")
            exit(0)
    


def print_ctxt_interval(c,logN,size,size2):
    m = c.decrypt(inplace=False)
    for i in range(size,size2+1):
        print(i,m[i])
        if (math.isnan(m[i].real)):
            print ("nan detected..stop")
            exit(0)
