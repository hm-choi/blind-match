package main

import (
	"encoding/json"

<<<<<<< HEAD
	"github.com/blind_match/api"
	"github.com/blind_match/blind_match"
	"github.com/gin-gonic/gin"
=======
	"github.com/gin-gonic/gin"
	"github.com/hmchoi/api"
	"github.com/hmchoi/blind_auth"
>>>>>>> 8576893 (Develop (#1))
)

func main() {
	CLUSTER_NUM, LogN, FV_SIZE, NUM_CTXT := 3, 13, 128, 8

	KEY_PATH := "/root/src/blind_auth_new/key/depth3"
	urls := map[int]string{
		0: "http://175.45.195.254:18888/send-ctxt/",
		1: "http://223.130.133.129:18888/send-ctxt/",
		2: "http://175.45.204.179:18888/send-ctxt/",
	}
	params, o, _ := blind_match.SetTestParams(KEY_PATH, LogN, FV_SIZE, NUM_CTXT)

	router := gin.Default()
	router.Use(api.CORSMiddleware())

	router.POST("/send-ctxt/face/128/13-5", api.CORSMiddleware(), func(c *gin.Context) {
		tmp_urls := map[int]string{}

		for i := 0; i < 3; i++ {
			tmp_urls[i] = urls[i] + "face/128/13-5"
		}

		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := api.Sent2Clusters(tmp_urls, o, params, body.Ctxt, CLUSTER_NUM)
		api.ReturnPostAPI(c, result_b, 1)
	})

	router.POST("/send-ctxt/face/128/13-4", api.CORSMiddleware(), func(c *gin.Context) {
		tmp_urls := map[int]string{}

		for i := 0; i < 3; i++ {
			tmp_urls[i] = urls[i] + "face/128/13-4"
		}

		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := api.Sent2Clusters(tmp_urls, o, params, body.Ctxt, CLUSTER_NUM)
		api.ReturnPostAPI(c, result_b, 1)
	})

	router.POST("/send-ctxt/face/128/13-3", api.CORSMiddleware(), func(c *gin.Context) {
		tmp_urls := map[int]string{}
		for i := 0; i < 3; i++ {
			tmp_urls[i] = urls[i] + "face/128/13-3"
		}

		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := api.Sent2Clusters(tmp_urls, o, params, body.Ctxt, CLUSTER_NUM)
		api.ReturnPostAPI(c, result_b, 1)
	})

	router.POST("/send-ctxt/face/128/13-2", api.CORSMiddleware(), func(c *gin.Context) {
		tmp_urls := map[int]string{}

		for i := 0; i < 3; i++ {
			tmp_urls[i] = urls[i] + "face/128/13-2"
		}

		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := api.Sent2Clusters(tmp_urls, o, params, body.Ctxt, CLUSTER_NUM)
		api.ReturnPostAPI(c, result_b, 1)
	})

	router.POST("/send-ctxt/face/128/13-1", api.CORSMiddleware(), func(c *gin.Context) {
		tmp_urls := map[int]string{}

		for i := 0; i < 3; i++ {
			tmp_urls[i] = urls[i] + "face/128/13-1"
		}

		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := api.Sent2Clusters(tmp_urls, o, params, body.Ctxt, CLUSTER_NUM)
		api.ReturnPostAPI(c, result_b, 1)
	})

	router.POST("/send-ctxt/face/128/13-0", api.CORSMiddleware(), func(c *gin.Context) {
		tmp_urls := map[int]string{}

		for i := 0; i < 3; i++ {
			tmp_urls[i] = urls[i] + "face/128/13-0"
		}

		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := api.Sent2Clusters(tmp_urls, o, params, body.Ctxt, CLUSTER_NUM)
		api.ReturnPostAPI(c, result_b, 1)
	})

	router.POST("/send-ctxt/face/64/13-0", api.CORSMiddleware(), func(c *gin.Context) {
		tmp_urls := map[int]string{}

		for i := 0; i < 3; i++ {
			tmp_urls[i] = urls[i] + "face/64/13-0"
		}

		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := api.Sent2Clusters(tmp_urls, o, params, body.Ctxt, CLUSTER_NUM)
		api.ReturnPostAPI(c, result_b, 1)
	})
	router.POST("/send-ctxt/face/64/13-1", api.CORSMiddleware(), func(c *gin.Context) {
		tmp_urls := map[int]string{}

		for i := 0; i < 3; i++ {
			tmp_urls[i] = urls[i] + "face/64/13-1"
		}

		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := api.Sent2Clusters(tmp_urls, o, params, body.Ctxt, CLUSTER_NUM)
		api.ReturnPostAPI(c, result_b, 1)
	})
	router.POST("/send-ctxt/face/64/13-2", api.CORSMiddleware(), func(c *gin.Context) {
		tmp_urls := map[int]string{}

		for i := 0; i < 3; i++ {
			tmp_urls[i] = urls[i] + "face/64/13-2"
		}

		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := api.Sent2Clusters(tmp_urls, o, params, body.Ctxt, CLUSTER_NUM)
		api.ReturnPostAPI(c, result_b, 1)
	})
	router.POST("/send-ctxt/face/64/13-3", api.CORSMiddleware(), func(c *gin.Context) {
		tmp_urls := map[int]string{}

		for i := 0; i < 3; i++ {
			tmp_urls[i] = urls[i] + "face/64/13-3"
		}

		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := api.Sent2Clusters(tmp_urls, o, params, body.Ctxt, CLUSTER_NUM)
		api.ReturnPostAPI(c, result_b, 1)
	})
	router.POST("/send-ctxt/face/64/13-4", api.CORSMiddleware(), func(c *gin.Context) {
		tmp_urls := map[int]string{}

		for i := 0; i < 3; i++ {
			tmp_urls[i] = urls[i] + "face/64/13-4"
		}

		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := api.Sent2Clusters(tmp_urls, o, params, body.Ctxt, CLUSTER_NUM)
		api.ReturnPostAPI(c, result_b, 1)
	})
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	router.POST("/send-ctxt/legacy/128", api.CORSMiddleware(), func(c *gin.Context) {
		tmp_urls := map[int]string{}

		for i := 0; i < 3; i++ {
			tmp_urls[i] = urls[i] + "legacy/128"
		}

		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := api.Sent2Clusters(tmp_urls, o, params, body.Ctxt, CLUSTER_NUM)
		api.ReturnPostAPI(c, result_b, 1)
	})

	router.POST("/send-ctxt/legacy/64", api.CORSMiddleware(), func(c *gin.Context) {
		tmp_urls := map[int]string{}

		for i := 0; i < 3; i++ {
			tmp_urls[i] = urls[i] + "legacy/64"
		}

		var body api.RequestBody
		err := json.NewDecoder(c.Request.Body).Decode(&body)
		blind_match.ReturnErr(err)
		result_b := api.Sent2Clusters(tmp_urls, o, params, body.Ctxt, CLUSTER_NUM)
		api.ReturnPostAPI(c, result_b, 1)
	})

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	router.Run(":18888")
}
