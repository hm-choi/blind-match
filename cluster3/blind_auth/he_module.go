package blind_auth

import (
	"fmt"
	"os"
	"strconv"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

type KeySet struct {
	Pk  *rlwe.PublicKey
	Sk  *rlwe.SecretKey
	Rlk *rlwe.RelinearizationKey
	Gks []*rlwe.GaloisKey
}

type ObjectSet struct {
	Ecd  *hefloat.Encoder
	Enc  *rlwe.Encryptor
	Dec  *rlwe.Decryptor
	Eval *hefloat.Evaluator
}

func GenKeyPair(params hefloat.Parameters) *KeySet {
	galEls := []uint64{params.GaloisElement(-32), params.GaloisElement(-64), params.GaloisElement(-128)}
	for i := 0; i < 9; i++ {
		galEls = append(galEls, params.GaloisElement(PowerOfTwo(i)))
	}
	for i := 0; i < 17; i++ {
		galEls = append(galEls, params.GaloisElement((-1)*i))
	}
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	gks := kgen.GenGaloisKeysNew(galEls, sk)
	keySet := &KeySet{}
	keySet.Pk, keySet.Sk, keySet.Rlk, keySet.Gks = pk, sk, rlk, gks

	return keySet
}

func GetKeyPair(params hefloat.Parameters, path string, logN int) *ObjectSet {
	pk := &rlwe.PublicKey{}
	sk := &rlwe.SecretKey{}
	rlk := &rlwe.RelinearizationKey{}

	pk_b, err := os.ReadFile(path + "/public_key/0")
	ReturnErr(err)
	ReturnErr(pk.UnmarshalBinary(pk_b))

	sk_b, err := os.ReadFile(path + "/secret_key/0")
	ReturnErr(err)
	ReturnErr(sk.UnmarshalBinary(sk_b))

	relin_b, err := os.ReadFile(path + "/relin_key/0")
	ReturnErr(err)
	ReturnErr(rlk.UnmarshalBinary(relin_b))

	gks := []*rlwe.GaloisKey{}
	for i := 0; i < 29; i++ {
		galkey := &rlwe.GaloisKey{}
		gk_b, err := os.ReadFile(path + "/galois_key/" + strconv.Itoa(i))
		ReturnErr(err)
		ReturnErr(galkey.UnmarshalBinary(gk_b))
		gks = append(gks, galkey)
	}
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	obs := &ObjectSet{}
	ecd := hefloat.NewEncoder(params)
	enc := hefloat.NewEncryptor(params, pk)
	dec := hefloat.NewDecryptor(params, sk)
	eval := hefloat.NewEvaluator(params, evk)

	obs.Ecd, obs.Enc, obs.Dec, obs.Eval = ecd, enc, dec, eval
	return obs
}

func GetCtxtDbs(path string, VEC_SIZE int, LOG_NUM_CTXT int, NUM_INPUT_CTXT int) [][]*rlwe.Ciphertext {
	ctxt_dbs := make([][]*rlwe.Ciphertext, VEC_SIZE)
	for i := 0; i < VEC_SIZE; i++ {
		ctxt_dbs[i] = make([]*rlwe.Ciphertext, NUM_INPUT_CTXT)
		for j := 0; j < NUM_INPUT_CTXT; j++ {
			ctxt := &rlwe.Ciphertext{}
			path_ := path + "/" + strconv.Itoa(LOG_NUM_CTXT) + "/" + strconv.Itoa(i) + "_" + strconv.Itoa(j)
			b, err := os.ReadFile(path_)
			ReturnErr(err)
			ctxt.UnmarshalBinary(b)
			ctxt_dbs[i][j] = ctxt
		}
	}
	return ctxt_dbs
}

func Input_Expansion_Param(n int, m int) int {
	result := 1
	if (n/m)%2 == 0 {
		result = -1
	}
	return result
}

func HE_Cossim(params hefloat.Parameters, o *ObjectSet, input_ct_b []byte, NUM_CTXT int, LOG_NUM_CTXT int, LOG_FV_SIZE int, masks [][]float64, ctxt_dbs [][]*rlwe.Ciphertext) []byte {
	NUM_INPUT_CTXT := PowerOfTwo(LOG_NUM_CTXT)
	FV_SIZE := PowerOfTwo(LOG_FV_SIZE)
	VEC_SIZE := FV_SIZE / NUM_INPUT_CTXT
	test := make([]float64, NUM_CTXT)
	for i := 0; i < len(test); i++ {
		if i%VEC_SIZE == 0 {
			test[i] = 1.0
		}
	}
	input_ctxt := &rlwe.Ciphertext{}
	input_ctxt.UnmarshalBinary(input_ct_b)

	input_ctxts := make([]*rlwe.Ciphertext, NUM_INPUT_CTXT)
	for i := 0; i < NUM_INPUT_CTXT; i++ {
		c, err := o.Eval.MulRelinNew(input_ctxt, masks[i])
		o.Eval.Rescale(c, c)
		ReturnErr(err)
		for j := 0; j < LOG_NUM_CTXT; j++ {
			tmp, _ := o.Eval.RotateNew(c, Input_Expansion_Param(i, PowerOfTwo(j))*PowerOfTwo(j+LOG_FV_SIZE-LOG_NUM_CTXT))
			o.Eval.Add(c, tmp, c)
		}
		input_ctxts[i] = c
	}

	result := &rlwe.Ciphertext{}
	for i := 0; i < len(ctxt_dbs); i++ {
		if len(ctxt_dbs[0]) == 0 {
			fmt.Println("")
		} else {
			c, err := o.Eval.MulRelinNew(input_ctxts[0], ctxt_dbs[i][0])
			ReturnErr(err)
			o.Eval.Rescale(c, c)

			for j := 1; j < len(ctxt_dbs[0]); j++ {
				tmp, err := o.Eval.MulRelinNew(input_ctxts[j], ctxt_dbs[i][j])
				o.Eval.Rescale(tmp, tmp)
				ReturnErr(err)
				o.Eval.Add(c, tmp, c)
			}
			//////////////////////////////////////////////////////////
			// For Compression methods
			//////////////////////////////////////////////////////////
			for k := 0; k < LOG_FV_SIZE-LOG_NUM_CTXT; k++ {
				tmp, err := o.Eval.RotateNew(c, PowerOfTwo(k))
				ReturnErr(err)
				o.Eval.Add(c, tmp, c)
			}
			o.Eval.MulRelin(c, test, c)
			if i == 0 {
				result = c
			} else {
				o.Eval.Rotate(c, (-1)*i, c)
				o.Eval.Add(result, c, result)
			}
		}
	}
	o.Eval.Rotate(result, (-2)*(VEC_SIZE/4), result)
	result_b, err := result.MarshalBinary()
	ReturnErr(err)
	return result_b
}

func SetTestParams(KEY_PATH string, LogN int, FV_SIZE int, NUM_CTXT int, VEC_SIZE int) [][]float64 {
	params := Blind_Auth_Param(LogN)
	o := GetKeyPair(params, KEY_PATH, LogN)
	masks := GenMaskVector(params, o, FV_SIZE, VEC_SIZE)
	return masks
}
