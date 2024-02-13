package main

import modules

func main() {
	params := Blind_Auth_Param(13) // 13 means the log of the number of slots in our CKKS parameter setting. In our scenario, we use LogN = 13.
	modules.GenKeyPair(params, true)

	result_vec := blind_auth.ReadCSV("Write down the path stored the set of feature vectors")
	BASE_PATH := "Write down the path to store the ciphertexts"
	Clusterset := []string{"Ciphertext path of cluster1", "Ciphertext path of cluster2", "Ciphertext path of cluster3"}
	NUM_OF_CLUSTERS := 3 // The number of clusters. In our scenario, we use three clusters.
	LOG_OF_FV_SIZE := 6 // Log of the feature vector size. You can change it.
	for s := 0; s < NUM_OF_CLUSTERS; s++ {
		for i := 0; i < LOG_OF_FV_SIZE; i++ {
			FV_SIZE = modules.PowerOfTwo(LOG_OF_FV_SIZE) // The feature vector size.
			LOG_NUM_CTXT := i
			NUM_INPUT_CTXT := modules.PowerOfTwo(LOG_NUM_CTXT)
			Log_vec := LOG_OF_FV_SIZE - i 
			NUM_CTXT := 2048 // The total number of feature vectors in a cluster. In our scenario, we store 2,048 feature vectors in a cluster, and there are three clusters. So, total 6,144 = 2048*3 feature vectors are recognized at the same time. 
			generated := modules.GenCtxtDbs(params, o, result_vec[s*NUM_CTXT*FV_SIZE:(s+1)*NUM_CTXT*FV_SIZE], modules.PowerOfTwo(Log_vec), 8192, FV_SIZE, NUM_INPUT_CTXT)

			for j := 0; j < len(generated); j++ {
				for k := 0; k < len(generated[0]); k++ {
					// Define the path to store the result ciphertexts.
					path_name := BASE_PATH + Clusterset[s] + strconv.Itoa(LOG_NUM_CTXT) + "/" + strconv.Itoa(j) + "_" + strconv.Itoa(k)
					b, err := generated[j][k].MarshalBinary()
					blind_auth.ReturnErr(err)
					os.WriteFile(path_name, b, 0644)
				}
			}
		}
	}
	fmt.Println("save finished!")
}
