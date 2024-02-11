package main

import modules

func main() {
	params := Blind_Auth_Param(13)
	modules.GenKeyPair(params, true)

	result_vec := blind_auth.ReadCSV("Write down the path stored the set of feature vectors")
	BASE_PATH := "Write down the path to store the ciphertexts"
	clusterset := []string{"Ciphertext path of cluster1", "Ciphertext path of cluster2", "Ciphertext path of cluster3"}
	for s := 0; s < 3; s++ {
		for i := 0; i < 6; i++ {
			FV_SIZE = 64 // The feature vector size.
			LOG_NUM_CTXT := i
			NUM_INPUT_CTXT := modules.PowerOfTwo(LOG_NUM_CTXT)
			Log_vec := 6 - i // if 
			NUM_CTXT := 2048
			generated := modules.GenCtxtDbs(params, o, result_vec[s*NUM_CTXT*FV_SIZE:(s+1)*NUM_CTXT*FV_SIZE], modules.PowerOfTwo(Log_vec), 8192, FV_SIZE, NUM_INPUT_CTXT)

			for j := 0; j < len(generated); j++ {
				for k := 0; k < len(generated[0]); k++ {
					path_name := BASE_PATH + clusterset[s] + strconv.Itoa(LOG_NUM_CTXT) + "/" + strconv.Itoa(j) + "_" + strconv.Itoa(k)
					b, err := generated[j][k].MarshalBinary()
					blind_auth.ReturnErr(err)
					os.WriteFile(path_name, b, 0644)
				}
			}
		}
	}
	fmt.Println("save finished!")
}
