package blind_auth

import (
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func ReturnErr(err error) {
	if err != nil {
		panic(err)
	}
}

func PowerOfTwo(n int) int {
	result := 1
	for i := 0; i < n; i++ {
		result *= 2
	}
	return result
}

func GenMaskVector(param hefloat.Parameters, o *ObjectSet, fv_size int, block_size int) [][]float64 {
	result := make([][]float64, fv_size/block_size)

	for i := 0; i < fv_size/block_size; i++ {
		tmp_vec := []float64{}
		tmp := make([]float64, fv_size)
		for j := 0; j < block_size; j++ {
			tmp[block_size*i+j] = 1.0
		}
		for j := 0; j < param.N()/fv_size; j++ {
			tmp_vec = append(tmp_vec, tmp...)
		}
		result[i] = tmp_vec
	}
	return result
}
