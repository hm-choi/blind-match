# Client Server

The client is used for acting according to the rules of the client.
- In this directory, we offer the ```blind_match_client.so``` library, which supports encryption and decryption of client's data.
- The parameter setting is the same as that of the server.
- Store the onnx file of the CNN model for extracting the feature vector of input biometric images in this path, and set up the path in the main.py.  

- The following procedures are supported in this scenario. 
  - a. Choose an image to match up with the database and load the CNN model.
  - b. Extract the feature vector via the stored CNN model.
  - c. Encrypt the feature vector using the API in the binary library ```blind_match_client.so```.
  - d. Send the ciphertext to the main server.
  - e. Receive the result ciphertext from the main server, then decrypt it and get the ID that the corresponding decrypted score is within the range of the threshold. The default setting is for choosing the score between (-0.99, 1.01) and choosing the maximum score for supporting Rank-1 accuracy.

  **[Note]** If you just use a single cluster when it is hard to set up multiple servers for testing. In this case, you can not use the main server. The client can directly communicate with the cluster server.