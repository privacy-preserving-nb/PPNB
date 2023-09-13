# Privacy-Preserving Naive Bayesian Classifier

The GPU version of HEaaN code is not available for public release as it is a proprietary asset of CryptoLab.

So, these codes are re-constructed by using HEaaN-SDK.

HEaaN-SDK is a package for data science/data analysis under homomorphic encryption using the HEaaN library.

We used the CPU version of the HEaaN-SDK, which is publicly available, to program in the same manner as described in our paper.

Costs such as communication and time might be different which is represented in the paper.

## Experimental Results

**GPU version of HEaaN**

- Experiment environment : AMD RYZEN 5950X CPU, NVIDIA Quadro RTX A6000 48GB GPU, and 128GB RAM using ubuntu 20.04LTS

|  | D1 | D2 | D3 | D4 |
| --- | --- | --- | --- | --- |
| Training Time(seconds) | 12.73 ± 0.87 | 12.87 ± 0.06 | 14.2 ± 0.11 | 9.97 ± 0.11 |
| Inference Time(milliseconds) | 835 ± 0.7 | 835 ± 0.3 | 838 ± 0.2 | 3180 ± 4.0 |

**CPU version of HEaaN (HEaaN-SDK)**

- Experiment environment : Same as GPU version. The experiments were conducted within a Docker container.

|  | D1 | D2 | D3 | D4 |
| --- | --- | --- | --- | --- |
| Training Time(seconds) | 77.19 ± 2.14 | 77.50 ± 2.74 | 85.49 ± 2.13 | 63.96 ± 1.69 |
| Inference Time(seconds) | 8.83 ± 0.12 | 8.65 ± 0.05 | 8.62 ± 0.05 | 35.57 ± 0.61 |

The accuracy of the CPU version code is identical to the accuracy of the reported in the paper.

|  | D1 | D2 | D3 | D4 |
| --- | --- | --- | --- | --- |
| Accuracy | 100% | 100% | 97.8% | 74.9% |

---

## Brief Explanation Of Codes

- [run.sh](http://run.sh/) : shell script for running the main
- NB_WMain.py : main code for training and classfy each data
- NB_WModule.py : functions for encrypt, train, inference
- NB_log.py : approximate logarithm function
- data_devide.py : devide the test data

## Key File Setting

keys
├─public_keypack
│  └─PK
└─secret_keypack

1. Create “keys” folder with this structure
2. Inside the “PK” folder, extract and place the PK1.tar and PK2.tar files.
3. Inside the “secret_keypac” folder, extract and place the scretekey.tar file.

## Procedure to Training and Inference the data

1. Install HEaaN.stat docker image (https://hub.docker.com/r/cryptolabinc/heaan-stat)
    
    ```bash
    docker pull cryptolabinc/heaan-stat:0.2.0-cpu-x86_64-avx512
    
    ```
    
2. Create docker container
    
    ```bash
    docker run -d -p 8888:8888 --name <container-id> cryptolabinc/heaan-stat:0.2.0-cpu-x86_64-avx512
    
    ```
    
3. Clone this repository in your docker container
4. According to the instructions in the Key File Setting, create a ‘keys’ directory
5. Devide the test data : change the directory name in 'data_devide.py’
    
    ```bash
    python3 data_devide.py
    
    ```
    
6. Run shell script