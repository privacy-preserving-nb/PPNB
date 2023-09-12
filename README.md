# Privacy-Preserving Naive Bayesian Classifier

This repository contains code and data for "Fully homomorphic privacy-preserving Naive Bayes machine learning and classification"

It is impossible to open the GPU version of HEaaN code to the public, these codes are re-constructed by using HEaaN-SDK.

HEaaN-SDK is a package for data science/data analysis under homomorphic encryption using the HEaaN library.

We used the CPU version of the HEaaN-SDK, which is publicly available, to program in the same manner as described in our paper.

Costs such as communication and time might be different which is represented in the paper.

---

## Brief explanation of codes

- run.sh : shell script for running the main
- NB_WMain.py : main code for training and classfy each data
- NB_WModule.py : functions for encrypt, train, inference
- NB_log.py : approximate logarithm function
- data_devide.py : devide the test data

## Procedure to training and inference the data

1. Install HEaaN.stat docker image (https://hub.docker.com/r/cryptolabinc/heaan-stat)
    
    ```bash
    docker pull cryptolabinc/heaan-stat:0.2.0-cpu-x86_64-avx512
    ```
    
2. Create docker container
    
    ```bash
    docker run -d -p 8888:8888 --name <container-id> cryptolabinc/heaan-stat:0.2.0-cpu-x86_64-avx512
    ```
    
3. Clone this repository in your docker container
4. Devide the test data : change the directory name in 'data_devide.py’
    
    ```bash
    python3 data_devide.py
    ```
    
5. Run shell script
    
    ```bash
    sh run.sh
    ```
