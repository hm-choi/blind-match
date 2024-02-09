# blind-match
## Blind-Match: Efficient Homomorphic Encryption-Based 1:N Matching for Privacy-Preserving Biometric Identification
![Overview of Blind-Match](images/Blind-Match.png)

## 1. Server Setting
- In this test, there are five servers are used.
  - A client, a main server, and three cluster servers.
  
- [Note] If you're not in a situation where you're going to have five servers, please prepare two servers: A client and a cluster (Use cluster1).

- We use NAVER Cloud servers (https://www.ncloud.com/product/compute/server)
  - All server spec are Standard-g2 Server.
  - CentOS 7.8.64
  - Two cores (Intel(R) Xeon(R) Gold 5220 CPU @ 2.20GHz) with 8GB DRAM

## 2. Installation
### Requirements
- OS: Linux
- Python: 3.9
  - aiohttp==3.8.4
  - Flask==2.2.3
  - numpy==1.24.2
  - onnx==1.15.0
  - onnxruntime==1.16.3
  - triton==2.1.0
- Lattigo V5 library (https://github.com/tuneinsight/lattigo)

## 3. Setting
### 3.1 Prepare Dataset
(1) Fingerprint Datasets
- PolyU Cross Sensor Fingerprint Database (PolyU)
  - It can be obtained by contacting the Hong Kong Polytechnic University.

- FVC(Fingerprint Verification Competition) Datasets
  - We use three versions (2000, 2002, 2004) of the FVC datasets in this experiment.
  - Each dataset can be obtained by the homepage of the Fingerprint Verification Competition.
  - FVC 2000 can be obtained in the website: http://bias.csr.unibo.it/fvc2000/download.asp
  - FVC 2002 can be obtained in the website: http://bias.csr.unibo.it/fvc2002/download.asp
  - FVC 2004 can be obtained in the website: http://bias.csr.unibo.it/fvc2004/download.asp

- CASIA V5 Datasets
  - It can be obtained by contacting the website of the Institute of Automation Chinese Academy of Sciences: http://english.ia.cas.cn/db/201611/t20161101_169922.html


(2) Face Datasets
### TBD

###  3.2 Data Preprocessing

### 3.3 Model Training
- We use ResNet-18 based CNN architecture for feature vector extraction.
- Detailed guide is in the ```training``` directory.

### 3.4. Client Setting

### 3.5 Main Server Setting

### 3.6 Cluster Server Setting

## 4. Run codes
- Run the Main server and three cluster servers.
- Run the Client code.
### How to use
See README.md files in each directories for guides.
