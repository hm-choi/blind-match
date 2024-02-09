package blind_auth

import (
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
)

func Blind_Auth_Param(logN int) hefloat.Parameters {
	result, _ := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            13,
		LogQ:            []int{33, 30, 30, 30},
		LogP:            []int{35},
		LogDefaultScale: 30,
		RingType:        ring.ConjugateInvariant,
	})
	return result
}
