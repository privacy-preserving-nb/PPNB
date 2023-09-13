# Privacy-Preserving Naive Bayesian Classifier

The GPU version of HEaaN code is not available for public release as it is a proprietary asset of CryptoLab. 

So, these codes are re-constructed by using HEaaN-SDK.

HEaaN-SDK is a package for data science/data analysis under homomorphic encryption using the HEaaN library.

We used the CPU version of the HEaaN-SDK, which is publicly available, to program in the same manner as described in our paper.

Costs such as communication and time might be different which is represented in the paper.

## Experimental results

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
| Training Time(seconds) | 93.23 | 75.43 | 84.24 | 60.67 |
| Inference Time(seconds) | 5.96 ± 0.1 | 8.65 ± 0.05 | 8.62 ± 0.05 |  35.57 ± 0.61 |

The accuracy of the CPU version code is identical to the accuracy of the reported in the paper.

|  | D1 | D2 | D3 | D4 |
| --- | --- | --- | --- | --- |
| Accuracy | 100% | 100% | 97.8% | 74.9% |

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
