import heaan_sdk as heaan
import os
import time
import numpy as np
import mpmath
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "16"  # set the number of CPU threads to use for parallel regions
import pandas as pd

# set key_dir_path
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


LOG2_HI = float.fromhex('0x1.62e42fee00000p-1')
LOG2_LOW = float.fromhex('0x1.a39ef35793c76p-33')
L1 = float.fromhex('0x1.5555555555593p-1')
L2 = float.fromhex('0x1.999999997fa04p-2')
L3 = float.fromhex('0x1.2492494229359p-2')
L4 = float.fromhex('0x1.c71c51d8e78afp-3')
L5 = float.fromhex('0x1.7466496cb03dep-3')
L6 = float.fromhex('0x1.39a09d078c69fp-3')
L7 = float.fromhex('0x1.2f112df3e5244p-3')
SQRT2_HALF = float.fromhex('0x1.6a09e667f3bcdp-1')
CTX = mpmath.MPContext()
CTX.prec = 200  # Bits vs. default of 53


##All1e1
l=[0]*(1<<log_num_slot)
for j in range((1<<log_num_slot)//25):
    for i in range(1,24):
        l[j*25+i]=1
    l[j*25]=0

he_all1e1 = heaan.Block(context, encrypted=False, data=l)


## Table
table = [0]*(1<<log_num_slot)
for j in range((1<<log_num_slot)//25):
    table[j*25+0] = -16.6366323334

he_table = heaan.Block(context, encrypted=False, data=table)


l=[0]*(1<<log_num_slot)
for j in range((1<<log_num_slot)//25):
    l[j*25+0]=l[j*25+1]=l[j*25+2]=1 
    for i in range(3,25):
        l[j*25+i]=0 


he_all0e3 = heaan.Block(context, encrypted=False, data=l)

l = [0]*(1 << log_num_slot)
for j in range((1<<log_num_slot)//25): 
    for i in range(24):
        l[j*25+i]=1

he_all1 = heaan.Block(context, encrypted=False, data=l)

l=[0]*(1<<log_num_slot)
for j in range((1<<log_num_slot)//25):
    l[j*25+0]=1 
    for i in range(1,25):
        l[j*25+i]=0 


he_all0e1 = heaan.Block(context, encrypted=False, data=l)

msg_pr = [0] * num_slot
for j in range(num_slot//25):
    for i in range(1,25):
        msg_pr[j*25+i] = 2**((-1)*(24-i))

he_msg_pr = heaan.Block(context, encrypted=False, data=msg_pr)

msg_pr2 = [0]*num_slot
for j in range(num_slot//25):
    for i in range(1,25):
        msg_pr2[j*25+i] = 2**(23-i)*1.4142135623730950488016887242

he_msg_pr2 = heaan.Block(context, encrypted=False, data=msg_pr2)

msg_pr3 = [0]*num_slot
for j in range(num_slot//25):
    for i in range(1,25):
        msg_pr3[j*25+i] = 0.693147180559945309417232121458*(23.5-i)

he_msg_pr3 = heaan.Block(context, encrypted=False, data=msg_pr3)


def log_approx(x):
    f = x - 1
    k = 0

    s = f / (2 + f)
    s2 = s * s
    s4 = s2 * s2
    # Terms with odd powers of s^2.
    t1 = s2 * (L1 + s4 * (L3 + s4 * (L5 + s4 * L7)))
    # Terms with even powers of s^2.
    t2 = s4 * (L2 + s4 * (L4 + s4 * L6))
    R = t1 + t2
    hfsq = 0.5 * f * f
    return k * LOG2_HI - ((hfsq - (s * (hfsq + R) + k * LOG2_LOW)) - f)

## Need make sure 2^(-1/2)~2^(1/2)
def HE_log_approx(ctxt, log_num_slot, context):
    check_boot(ctxt)

    ctxt_x1 = ctxt + 1
    ctxt_x1_inv = ctxt_x1.inverse(greater_than_one = True, init = 0.1, num_iter=16, inplace=False) 
    check_boot(ctxt_x1_inv)
  
    ctxt_x1 = ctxt_x1 - 2

    s = ctxt_x1_inv * ctxt_x1
    check_boot(s)

    s2 = s*s
    check_boot(s2)
    
    s4 = s2*s2
    check_boot(s4)

    tmp = s4 * L7
    check_boot(tmp)
    
    tmp = tmp + L5
    
    tmp = tmp * s4
    check_boot(tmp)

    tmp = tmp + L3

    tmp = tmp * s4
    check_boot(tmp)

    tmp = tmp + L1

    tmp = tmp *s2
    check_boot(tmp)

    tmp2= s4*L6
    check_boot(tmp2)

    tmp2 = tmp2+L4

    tmp2 = tmp2*s4
    check_boot(tmp2)

    tmp2 = tmp2 + L2
    

    tmp2= tmp2*s4
    check_boot(tmp2)

    R = tmp + tmp2
  
    ctxt_x1_inv = ctxt_x1*ctxt_x1## ctxt_x1_inv = hfsq
    check_boot(ctxt_x1_inv)
    

    ctxt_x1_inv = ctxt_x1_inv*0.5
    check_boot(ctxt_x1_inv)

    tmp = ctxt_x1 - ctxt_x1_inv ## tmp = f- hsfq
    tmp2 = ctxt_x1_inv + R ## tmp 2 = hfsq + R
    
    tmp2 = tmp2 * s ## tmp2 = s*(hfsq+R)
    check_boot(tmp2)

    tmp = tmp + tmp2 ## f-hsfq + s*(hfsq+R) 
    check_boot(tmp)

    return tmp


def check_boot(x):
    if x.level==3:
        x.bootstrap()
    elif x.level<3:
        print("ciphertext level is less than 3.. exiting..\n")
        exit(1)
    return

## Suppose every set of 25 slots has the same values to be logged
def find_log(ctxt1, log_num_slot, context):
    
    ctxt = ctxt1
    
    ctxt = ctxt.__neg__()
    check_boot(ctxt)
    

    cf = ctxt + he_msg_pr
    check_boot(cf)

    ctxt = ctxt.__neg__()
    check_boot(ctxt)
    
 
    cf.sign(inplace = True, log_range=0)
    check_boot(cf)

    cf1 = cf.__lshift__(1)
    check_boot(cf1)

    cf = cf + cf1

    cf = cf * 0.5
    check_boot(cf)

    cf = cf * cf
    check_boot(cf)
    ## 1-()
    ## all are 1s

    cf = cf.__neg__()

    cf = cf + he_all1

    ctxt = ctxt * cf
    check_boot(ctxt)

    ctxt = ctxt * he_msg_pr2
    check_boot(ctxt)


    ctxt.bootstrap()
    
    ctxtr = HE_log_approx(ctxt, log_num_slot, context)
    check_boot(ctxtr)

    ctxtr = ctxtr-he_msg_pr3


    ctxtr = ctxtr * cf
    check_boot(ctxtr)
    ## All 1 except 1st

    ctxtr = ctxtr * he_all1e1
    check_boot(ctxtr)

    tmp2 = cf * he_table
    check_boot(tmp2)

    ctxtr = ctxtr + tmp2
    check_boot(ctxtr)

    ### Sum up all the values per 25 slots
    
    tmp = ctxtr.__lshift__(12)  
    check_boot(tmp)
    
    ctxtr = ctxtr+tmp
    check_boot(ctxtr)
    
    tmp = ctxtr.__lshift__(6)
    
    check_boot(tmp)
   
    ctxtr = ctxtr + tmp
    check_boot(ctxtr)
    
    
    
    tmp = ctxtr.__lshift__(3)    
    check_boot(tmp)
    
    ctxtr = ctxtr + tmp
    check_boot(ctxtr)

    ## All except 1st three

    ctxtr = ctxtr * he_all0e3
    check_boot(ctxtr)

    tmp = ctxtr.__lshift__(2)     
    check_boot(tmp)

    ctxtr = ctxtr + tmp
    
    tmp = ctxtr.__lshift__(1)    
    check_boot(tmp)
    

    ctxtr = ctxtr + tmp
    
    ## All zero except 1st

    ctxtr = ctxtr * he_all0e1
    check_boot(ctxtr)
 
    ### Sum up all the values per 24 slots
    tmp = cf.__lshift__(12)    
    check_boot(tmp)
    cf = cf + tmp

    tmp = cf.__lshift__(6)      
    check_boot(tmp)
    cf = cf +tmp
    
    tmp = cf.__lshift__(3)    
    check_boot(tmp)
    cf = cf +tmp
   
    ## All zero except 1st three
  
    cf = cf * he_all0e3
    check_boot(cf)

    tmp = cf.__lshift__(2)   
    check_boot(tmp)
    cf = cf + tmp

    tmp = cf.__lshift__(1)    
    check_boot(tmp)
    cf = cf + tmp
   
    ## All zero except 1st

    cf = cf * he_all0e1
    check_boot(cf)

    cf.inverse(greater_than_one = True, init = 0.1, num_iter=9, inplace=True) 
    
    check_boot(cf)

    ctxtr = ctxtr * cf
    check_boot(ctxtr) 
    return ctxtr

