package api

import (
	"net/http"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
)

type RequestBody struct {
	LogSlots int
	Ctxt     []byte `json:"ctxt"`
}

type Response struct {
	Message string `json:"message"`
	Data    string `json:"data"`
}
type ResponseBody struct {
	Result []byte `json:"data"`
}

type HttpResp struct {
	Id   string
	Resp *http.Response
	Err  error
}

type Resp struct {
	Data  *rlwe.Ciphertext
	Index int
}
