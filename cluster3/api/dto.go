package api

type RequestBody struct {
	LogSlots int
	Ctxt     []byte `json:"ctxt"`
}

type ResponseBody struct {
	Result []byte `json:"data"`
}
