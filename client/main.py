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

KEY_PATH = "/root/src/blind_auth_new/key/depth3/"

FV_SIZE = 32
R_FV_SIZE = FV_SIZE * 4
LOG_NUM_CTXT = 2

NUM_CTXT =  2**  LOG_NUM_CTXT
VEC_SIZE = FV_SIZE/NUM_CTXT
R_VEC_SIZE = R_FV_SIZE / NUM_CTXT
print('R_FV_SIZE', R_FV_SIZE, 'R_VEC_SIZE', R_VEC_SIZE)
URLs = ['http://223.130.132.70:18888/send-ctxt/face/' + str(R_FV_SIZE) + '/13-' + str(LOG_NUM_CTXT)]
# URLs = ['http://175.45.195.254:18888/send-ctxt/face/' + str(R_FV_SIZE) + '/13-' + str(LOG_NUM_CTXT)]
# URLs = ['http://223.130.132.70:18888/send-ctxt/legacy/64']

NUM_SLOTS = 2048
R_NUM_SLOTS = 2048*4
CLUSTER_NUM = 3
II = 0
def main():
    RequestBody = {
        'LogSlots':13,
        'Ctxt':1
    } 
    INDEX = random.randint(NUM_SLOTS*II,NUM_SLOTS*(II+1))
    f = open('/root/src/dataset/r18_feat'+ str(R_FV_SIZE) +'_ms1mv3.csv', 'r')
    rdr = csv.reader(f)
    line_ = []
    for line in rdr:
        line_ = line[R_FV_SIZE*INDEX:R_FV_SIZE*(INDEX+1)]
    f.close()     
        
    ort_sess = ort.InferenceSession('/root/src/model/r18_' + str(R_FV_SIZE) + '.onnx')
    x = np.array([np.float32(0.1) for _ in range(3*112*112)])
    x = x.reshape([1,3,112,112])
    START_TIME = time.time()
    outputs_ = ort_sess.run(None, {'input': x})
    outputs_ = outputs_[0][0]

    print('ONNX Inference time', time.time()-START_TIME)
    line_ = line[R_FV_SIZE*INDEX:R_FV_SIZE*(INDEX+1)]
    outputs_ = []
    for i in range(R_FV_SIZE):
        outputs_.append(float(line_[i])) 
 
    outputs = []
    for _ in range(int(8192//len(outputs_))):
        for j in range(len(outputs_)):
            outputs.append(outputs_[j])
     


    c_float_array = (c_double * len(outputs))(*outputs)
    key_string_c = ctypes.c_char_p(KEY_PATH.encode('utf-8'))
    encrypt = lib.encrypt
    ENCRYPTION_TIME = time.time()
    result = encrypt(c_int(13), key_string_c, c_int(len(outputs)), c_float_array)
    print('Encryption Time', time.time() - ENCRYPTION_TIME)
    
    message_from_go = ctypes.c_char_p(result).value.decode("utf-8")
    RequestBody['Ctxt'] = message_from_go


    decrypt = lib.decrypt
    decrypt.restype = c_void_p

    responses = asyncio.run(fetch_all(URLs, RequestBody))
    for i in range(len(responses[0])):
        json_object = json.loads(responses[0][i])

        decrypt.restype = ctypes.POINTER(ctypes.c_double)
        DECRYPTION_TIME = time.time()
        decrypted = decrypt(c_int(13), key_string_c, ctypes.c_char_p(json_object['data'].encode('utf-8')))
        print('Decryption Time', time.time() - DECRYPTION_TIME)
 
        # print("decrypted[0:10]", decrypted[0:10])
        clusterIdx = json_object['clusterIdx']
        for i in range(0,R_NUM_SLOTS):
            if decrypted[i] > 0.99 and decrypted[i] < 1.1:
                PREDICTED_VAL = i
                PREDICTED = int((i // R_VEC_SIZE) + (i%R_VEC_SIZE)*(8192//R_VEC_SIZE))
                print(INDEX, PREDICTED, decrypted[PREDICTED_VAL])
        # for i in range(24):
        #     print(i, "decrypted[i]", decrypted[i])
    print('Final time', time.time()-START_TIME)




# def main1():

#     KEY_PATH = "/root/src/keyset/depth3/"
#     x = np.array([np.float32(0.1) for _ in range(3*112*112)])
#     x = x.reshape([1,3,112,112])
#     ort_sess = ort.InferenceSession('/root/src/model/r18_128.onnx')
#     outputs = ort_sess.run(None, {'input': x})
#     outputs = outputs[0][0]
#     print("outputs[:10]", outputs[:10])


#     encrypt = lib.encrypt
#     encrypt.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_char_p,ctypes.c_int]
   
#     encrypt.restype = ctypes.POINTER(ctypes.c_ubyte * 131419)  # 여기에서는 16을 반환하는 예제로 가정
    
#     c_float_array = outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#     c_array_length = ctypes.c_int(len(outputs))
#     C_logN = ctypes.c_int(11)
#     c_string = ctypes.c_char_p(KEY_PATH.encode('utf-8'))

#     result = encrypt(c_float_array, c_array_length, c_string, C_logN).contents
   
    # byte_slice = bytes(result[:])
    # print(type(result))

    # c_float_array = result.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte * 131419))
    # decrypt = lib.decrypt
    # decrypt.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_char_p,ctypes.c_int]
    # output = decrypt(result, 131419, c_string, 11)
    # print(type(output))

      
 

if __name__ == "__main__":
    main()


