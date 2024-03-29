# Main Server

The main server is used for acting the role of the main node.
- The following procedures are supported in this scenario. 
  - a. When receives the client's HTTP(POST) requests, then it sends the requests to each cluster server.
  - b. It receives each response from clusters and then combines each ciphertext in the cluster's HTTP response body as a ciphertext. 
  - c. It returned the combined ciphertext to the client via the HTTP response body.

**[Note]** If you just use a single cluster when it is hard to set up multiple servers for testing. In this case, you can not use the main server. The client can directly communicate with the cluster server.