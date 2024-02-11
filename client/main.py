import ctypes
import random
import csv
import json
import onnxruntime as ort
import numpy as np
from ctypes import CDLL, c_double, c_int, POINTER, c_void_p
import time
import asyncio
import aiohttp

lib = ctypes.cdll.LoadLibrary('./blind_match_client.so')
async def fetch(session, url, requestBody):
    async with session.post(url, json=requestBody) as response:
        return await response.text()

async def fetch_all(urls, requestBody):
    response = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(len(urls)):
            url = urls[i]
            task = asyncio.ensure_future(fetch(session, url, requestBody))
            tasks.append(task)
        responses = await asyncio.gather(*tasks)  
        response.append(responses)
        
    return response

"""
- KEY_PATH: Set up the path where the key pair is stored.
- DATA_PATH: Set up the path where the input data is stored.
- MODEL_PATH: Set up the path where the CNN model is stored.
- R_FV_SIZE: The feature vector size for the test. If you want to test for the model with feature vector size = 128, 
Setup the parameter R_FV_SIZE = 128.
- FV_SIZE: The multiplication result between R_FV_SIZE and REAL_NUM_CTXT/NUM_SLOTS. For example, in our default experiment, 
we test 
- LOG_NUM_INPUT_CTXT: The log of the number of input ciphertext expansions (Algorithm 2 in the paper.) 
- VEC_SIZE: The division of FV_SIZE and NUM_CTXT.
- NUM_STORED_CTXT: The total number of encrypted feature vectors in each cluster. In our scenario, we store 2,048 feature vectors in each cluster.
- NUM_SLOTS: The number of slots in the CKKS scheme. We use 8,192 as the number of slots in our scenario.
"""
KEY_PATH = "/root/src/blind_auth_new/key/depth3/"
DATA_PATH = "/root/src/dataset/r18_feat128_ms1mv3.csv"
MODEL_PATH = "/root/src/model/r18_128.onnx"
R_FV_SIZE = 128
FV_SIZE = R_FV_SIZE / 4
LOG_NUM_INPUT_CTXT = 2 
NUM_CTXT =  2** LOG_NUM_INPUT_CTXT
VEC_SIZE = FV_SIZE/NUM_CTXT
R_VEC_SIZE = R_FV_SIZE / NUM_CTXT
NUM_STORED_CTXT = 2048
NUM_SLOTS = 8192
MAIN_SERVER_INFO = 'Write down the IP and Port of the main server (e.g, XXX.XXX.XXX.XXX:YYYY)'
"""
Set up the main server's URL.
- Write down the main server's URL.
- If you want to one cluster server, then write down the cluster's URL instead of the main server's URL.
"""
URLs = ['http://' + MAIN_SERVER_INFO + '/send-ctxt/face/' + str(R_FV_SIZE) + '/13-' + str(LOG_NUM_INPUT_CTXT)]

II = 0
def main():
    ## =============== Library Setting =============== ##
    # Setup the library for encryption and decryption.
    ## =============================================== ##
    encrypt = lib.encrypt 
    decrypt = lib.decrypt
    decrypt.restype = c_void_p
    RequestBody = {
        'LogSlots':13,
        'Ctxt':1
    } 

    """
    a. Choose an image and load the CNN model.
    - The input image can be extracted by using various implementation methods. 
      - Only fix the input size as 3 X 112 X 112.
    - Choose an index in the range between 0 and the total number of ciphertext for the test. 
      - You can choose the INDEX manually.
    """
    #######################################################################################
    # Load the input biometric image. The detail of the code needs to be implemented.
    #######################################################################################
    
    # Choose an index of the input image for the test.
    INDEX = random.randint(0, 6144)

     # Load the data
    f = open(DATA_PATH, 'r')
    rdr = csv.reader(f)
    line_ = []
    for line in rdr:
        line_ = line[R_FV_SIZE*INDEX:R_FV_SIZE*(INDEX+1)]
    f.close()     
        
    # Extract the CNN model from the onnx file.
    # If you want to customize this part, fix the following code. 
    # e.g., if you want to use the CNN model directly, then the onnx inference do not need.
    ort_sess = ort.InferenceSession(MODEL_PATH)
    

    """
    b. Extract the feature vector via the stored CNN model.
    - Firstly, choose the feature vector of the input image
    - 
    """
    outputs_ = ort_sess.run(None, {'input': x})
    outputs_ = outputs_[0][0]
    line_ = line[R_FV_SIZE*INDEX:R_FV_SIZE*(INDEX+1)]
    outputs_ = []
    for i in range(R_FV_SIZE):
        outputs_.append(float(line_[i])) 
    
    # Duplicate the input feature vector before encryption.
    # For batch processing, we duplicate the feature vector as in Figure 4 (C_u) in our paper.
    outputs = []
    for _ in range(int(NUM_SLOTS//len(outputs_))):
        for j in range(len(outputs_)):
            outputs.append(outputs_[j])
     

    """
    c. Encrypt the feature vector
    """
    c_float_array = (c_double * len(outputs))(*outputs)
    key_string_c = ctypes.c_char_p(KEY_PATH.encode('utf-8'))
    result = encrypt(c_int(13), key_string_c, c_int(len(outputs)), c_float_array)

    """
    d. Send the ciphertext to the main server.
    """
    message_from_go = ctypes.c_char_p(result).value.decode("utf-8")
    RequestBody['Ctxt'] = message_from_go

    """
    e. Receive the result ciphertext and decrypt it.
    - First, receive the result ciphertext from the server 
    - Next, decrypt it and get the ID of the decrypted result.
    """
    responses = asyncio.run(fetch_all(URLs, RequestBody))
    for i in range(len(responses[0])):
        json_object = json.loads(responses[0][i])
        # Decryption process
        decrypt.restype = ctypes.POINTER(ctypes.c_double)
        decrypted = decrypt(c_int(13), key_string_c, ctypes.c_char_p(json_object['data'].encode('utf-8')))
        for i in range(0,NUM_SLOTS):
            if decrypted[i] > 0.99 and decrypted[i] < 1.1:
                PREDICTED_VAL = i
                PREDICTED = int((i // R_VEC_SIZE) + (i%R_VEC_SIZE)*(NUM_SLOTS//R_VEC_SIZE))
                print(INDEX, PREDICTED, decrypted[PREDICTED_VAL])

if __name__ == "__main__":
    main()


