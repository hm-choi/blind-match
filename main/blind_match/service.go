package blind_auth

import (
	"os"
	"strconv"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

type ObjectSet struct {
	Ecd  *hefloat.Encoder
	Enc  *rlwe.Encryptor
	Dec  *rlwe.Decryptor
	Eval *hefloat.Evaluator
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

func SetTestParams(KEY_PATH string, LogN int, FV_SIZE int, NUM_CTXT int) (hefloat.Parameters, *ObjectSet, [][]float64) {
	VEC_SIZE := FV_SIZE / NUM_CTXT
	params := Blind_Auth_Param(LogN)
	o := GetKeyPair(params, KEY_PATH, LogN)
	masks := GenMaskVector(params, o, FV_SIZE, VEC_SIZE)
	return params, o, masks
}
