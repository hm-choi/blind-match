package modules

import (
	"os"

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

func GenKeyPair(params hefloat.Parameters, StoreTF bool) *KeySet {
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

	if StoreTF {
		sk_b, _ := sk.MarshalBinary()
		os.WriteFile("Write your secret key path", sk_b, 0644)

		pk_b, _ := pk.MarshalBinary()
		os.WriteFile("Write your public key path", pk_b, 0644)

		rlk_b, _ := rlk.MarshalBinary()
		os.WriteFile("Write your re-linear key path", rlk_b, 0644)

		for i := 0; i < len(gks); i++ {
			gk_b, _ := gks[i].MarshalBinary()
			os.WriteFile("Write your galois key path", gk_b, 0644)
		}
	}
	keySet := &KeySet{}
	keySet.Pk, keySet.Sk, keySet.Rlk, keySet.Gks = pk, sk, rlk, gks
	return keySet
}

func GenCtxtDbs(params hefloat.Parameters, o *ObjectSet, result_vec []float64, VEC_SIZE int, NUM_CTXT int, FV_SIZE int, NUM_INPUT_CTXT int) [][]*rlwe.Ciphertext {
	ctxt_dbs := make([][]*rlwe.Ciphertext, VEC_SIZE)
	plain_dbs := make([][]float64, VEC_SIZE)
	for i := 0; i < VEC_SIZE; i++ {
		plain_dbs[i] = result_vec[NUM_CTXT*NUM_INPUT_CTXT*i : NUM_CTXT*NUM_INPUT_CTXT*(i+1)]
	}
	for k := 0; k < VEC_SIZE; k++ {
		ctxt_dbs[k] = make([]*rlwe.Ciphertext, NUM_INPUT_CTXT)
		for i := 0; i < NUM_INPUT_CTXT; i++ {
			tmp := []float64{}
			for j := 0; j < NUM_CTXT/VEC_SIZE; j++ {
				tmp = append(tmp, plain_dbs[k][FV_SIZE*j+VEC_SIZE*i:FV_SIZE*j+VEC_SIZE*(i+1)]...)
			}
			pt := hefloat.NewPlaintext(params, params.MaxLevel())
			o.Ecd.Encode(tmp, pt)
			ct, err := o.Enc.EncryptNew(pt)
			ReturnErr(err)
			ctxt_dbs[k][i] = ct
		}
	}
	return ctxt_dbs
}
