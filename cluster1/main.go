package main

import (
	"encoding/json"
	"fmt"
	"time"

	api "github.com/blind_match/blind_api"
	"github.com/blind_match/blind_match"
	"github.com/gin-gonic/gin"
)

type Response struct {
	Message    string `json:"message"`
	Data       string `json:"data"`
	clusterIdx int
}

func main() {
	router := gin.Default()
	router.Use(api.CORSMiddleware())

	CLUSTER_IDX := 0

	// LogN means the degree parameter of CKKS. In our setting, LogN = 13
	LogN := 13
	NUM_CTXT := blind_match.PowerOfTwo(LogN)

	// KEY_PATH is the set of path of the stored key set.
	// CTXT_PATH_128 is the path that the saved ciphertext with feature vector size 128 in the server
	// CTXT_PATH_64 is the path that the saved ciphertext with feature vector size 64 in the server
	KEY_PATH := "/root/src/blind_auth_new/key/depth3"
	CTXT_PATH_128 := "/root/src/blind_auth_new/ctxt/cluster1/r18_feat128_ms1mv3/"
	CTXT_PATH_64 := "/root/src/blind_auth_new/ctxt/cluster1/r18_feat64_ms1mv3/"

	params := blind_match.Blind_Auth_Param(LogN)
	o := blind_match.GetKeyPair(params, KEY_PATH, LogN)

	ctxt_dbs_128_0 := blind_match.GetCtxtDbs(CTXT_PATH_128, 32, 0, 1)
	ctxt_dbs_128_1 := blind_match.GetCtxtDbs(CTXT_PATH_128, 16, 1, 2)
	ctxt_dbs_128_2 := blind_match.GetCtxtDbs(CTXT_PATH_128, 8, 2, 4)
	ctxt_dbs_128_3 := blind_match.GetCtxtDbs(CTXT_PATH_128, 4, 3, 8)
	ctxt_dbs_128_4 := blind_match.GetCtxtDbs(CTXT_PATH_128, 2, 4, 16)
	ctxt_dbs_128_5 := blind_match.GetCtxtDbs(CTXT_PATH_128, 1, 5, 32)

	masks_64_0 := blind_match.SetTestParams(KEY_PATH, LogN, 64, 1, 64)
	masks_64_1 := blind_match.SetTestParams(KEY_PATH, LogN, 64, 2, 32)
	masks_64_2 := blind_match.SetTestParams(KEY_PATH, LogN, 64, 4, 16)
	masks_64_3 := blind_match.SetTestParams(KEY_PATH, LogN, 64, 8, 8)
	masks_64_4 := blind_match.SetTestParams(KEY_PATH, LogN, 64, 16, 4)

	ctxt_dbs_64_0 := blind_match.GetCtxtDbs(CTXT_PATH_64, 16, 0, 1)
	ctxt_dbs_64_1 := blind_match.GetCtxtDbs(CTXT_PATH_64, 8, 1, 2)
	ctxt_dbs_64_2 := blind_match.GetCtxtDbs(CTXT_PATH_64, 4, 2, 4)
	ctxt_dbs_64_3 := blind_match.GetCtxtDbs(CTXT_PATH_64, 2, 3, 8)
	ctxt_dbs_64_4 := blind_match.GetCtxtDbs(CTXT_PATH_64, 1, 4, 16)

	masks_128_0 := blind_match.SetTestParams(KEY_PATH, LogN, 128, 1, 128)
	masks_128_1 := blind_match.SetTestParams(KEY_PATH, LogN, 128, 2, 64)
	masks_128_2 := blind_match.SetTestParams(KEY_PATH, LogN, 128, 4, 32)
	masks_128_3 := blind_match.SetTestParams(KEY_PATH, LogN, 128, 8, 16)
	masks_128_4 := blind_match.SetTestParams(KEY_PATH, LogN, 128, 16, 8)
	masks_128_5 := blind_match.SetTestParams(KEY_PATH, LogN, 128, 32, 4)

	// ======================================================== //
	// Define API
	// ======================================================== //

	// ======================================================== //
	// Api for feature vector size = 128
	// ======================================================== //
	router.POST("/send-ctxt/face/128/13-0", api.CORSMiddleware(), func(c *gin.Context) {
		START_TIME := time.Now()
		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := blind_match.HE_Cossim(params, o, body.Ctxt, NUM_CTXT, 0, 7, masks_128_0, ctxt_dbs_128_0)
		fmt.Println("fv: 128, #ctxt: 4", time.Since(START_TIME))
		api.ReturnPostAPI(c, result_b, CLUSTER_IDX)
	})
	router.POST("/send-ctxt/face/128/13-1", api.CORSMiddleware(), func(c *gin.Context) {
		START_TIME := time.Now()
		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := blind_match.HE_Cossim(params, o, body.Ctxt, NUM_CTXT, 1, 7, masks_128_1, ctxt_dbs_128_1)
		fmt.Println("fv: 128, #ctxt: 4", time.Since(START_TIME))
		api.ReturnPostAPI(c, result_b, CLUSTER_IDX)
	})
	router.POST("/send-ctxt/face/128/13-2", api.CORSMiddleware(), func(c *gin.Context) {
		START_TIME := time.Now()
		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := blind_match.HE_Cossim(params, o, body.Ctxt, NUM_CTXT, 2, 7, masks_128_2, ctxt_dbs_128_2)
		fmt.Println("fv: 128, #ctxt: 4", time.Since(START_TIME))
		api.ReturnPostAPI(c, result_b, CLUSTER_IDX)
	})
	router.POST("/send-ctxt/face/128/13-3", api.CORSMiddleware(), func(c *gin.Context) {
		START_TIME := time.Now()
		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := blind_match.HE_Cossim(params, o, body.Ctxt, NUM_CTXT, 3, 7, masks_128_3, ctxt_dbs_128_3)
		fmt.Println("fv: 128, #ctxt: 8", time.Since(START_TIME))
		api.ReturnPostAPI(c, result_b, CLUSTER_IDX)
	})

	router.POST("/send-ctxt/face/128/13-4", api.CORSMiddleware(), func(c *gin.Context) {
		START_TIME := time.Now()
		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		input_ct_b := body.Ctxt
		result_b := blind_match.HE_Cossim(params, o, input_ct_b, NUM_CTXT, 4, 7, masks_128_4, ctxt_dbs_128_4)
		fmt.Println("fv: 128, #ctxt: 16", time.Since(START_TIME))
		api.ReturnPostAPI(c, result_b, CLUSTER_IDX)
	})

	router.POST("/send-ctxt/face/128/13-5", api.CORSMiddleware(), func(c *gin.Context) {
		START_TIME := time.Now()
		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := blind_match.HE_Cossim(params, o, body.Ctxt, NUM_CTXT, 5, 7, masks_128_5, ctxt_dbs_128_5)
		fmt.Println("fv: 128, #ctxt: 32", time.Since(START_TIME))
		api.ReturnPostAPI(c, result_b, CLUSTER_IDX)
	})

	// ======================================================== //
	// Api for feature vector size = 64
	// ======================================================== //
	router.POST("/send-ctxt/face/64/13-0", api.CORSMiddleware(), func(c *gin.Context) {
		START_TIME := time.Now()
		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := blind_match.HE_Cossim(params, o, body.Ctxt, NUM_CTXT, 0, 6, masks_64_0, ctxt_dbs_64_0)
		fmt.Println("fv: 64, #ctxt: 4", time.Since(START_TIME))
		api.ReturnPostAPI(c, result_b, CLUSTER_IDX)
	})
	router.POST("/send-ctxt/face/64/13-1", api.CORSMiddleware(), func(c *gin.Context) {
		START_TIME := time.Now()
		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := blind_match.HE_Cossim(params, o, body.Ctxt, NUM_CTXT, 1, 6, masks_64_1, ctxt_dbs_64_1)
		fmt.Println("fv: 64, #ctxt: 1", time.Since(START_TIME))
		api.ReturnPostAPI(c, result_b, CLUSTER_IDX)
	})
	router.POST("/send-ctxt/face/64/13-2", api.CORSMiddleware(), func(c *gin.Context) {
		START_TIME := time.Now()
		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := blind_match.HE_Cossim(params, o, body.Ctxt, NUM_CTXT, 2, 6, masks_64_2, ctxt_dbs_64_2)
		fmt.Println("fv: 64, #ctxt: 2", time.Since(START_TIME))
		api.ReturnPostAPI(c, result_b, CLUSTER_IDX)
	})

	router.POST("/send-ctxt/face/64/13-3", api.CORSMiddleware(), func(c *gin.Context) {
		START_TIME := time.Now()
		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := blind_match.HE_Cossim(params, o, body.Ctxt, NUM_CTXT, 3, 6, masks_64_3, ctxt_dbs_64_3)
		fmt.Println("fv: 64, #ctxt: 8", time.Since(START_TIME))
		api.ReturnPostAPI(c, result_b, CLUSTER_IDX)
	})
	router.POST("/send-ctxt/face/64/13-4", api.CORSMiddleware(), func(c *gin.Context) {
		START_TIME := time.Now()
		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := blind_match.HE_Cossim(params, o, body.Ctxt, NUM_CTXT, 4, 6, masks_64_4, ctxt_dbs_64_4)
		fmt.Println("fv: 64, #ctxt: 16", time.Since(START_TIME))
		api.ReturnPostAPI(c, result_b, CLUSTER_IDX)
	})

	router.Run(":18888")
}
