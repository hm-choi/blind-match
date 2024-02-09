package api

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"sync"

	"github.com/gin-gonic/gin"
	"github.com/hmchoi/blind_auth"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

// CORSMiddleware handles CORS requests.
func CORSMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Credentials", "true")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, accept, origin, Cache-Control, X-Requested-With")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS, GET, PUT, DELETE")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	}
}

func ReturnPostAPI(c *gin.Context, result_b []byte, CLUSTER_IDX int) {
	c.JSON(http.StatusOK, gin.H{
		"message":    "ok",
		"data":       base64.StdEncoding.EncodeToString(result_b),
		"clusterIdx": CLUSTER_IDX,
	})
}

func ObjectMapper(c *gin.Context) RequestBody {
	var body RequestBody
	err := json.NewDecoder(c.Request.Body).Decode(&body)
	if err != nil {
		panic(err)
	}
	return body
}

func DoReq(url string, data *bytes.Buffer) []byte {
	resp, err := http.Post(url, "application/json", data)
	blind_auth.ReturnErr(err)
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	blind_auth.ReturnErr(err)

	return body
}

func Sent2Clusters(urls map[int]string, o *blind_auth.ObjectSet, params hefloat.Parameters, input_b []byte, CLUSTER_NUM int) []byte {
	var wg sync.WaitGroup
	queue := make(chan Resp, CLUSTER_NUM)
	var results []Resp

	for index, u := range urls {
		wg.Add(1)
		go func(url string, i int) {
			defer wg.Done()
			reqBody := RequestBody{}
			reqBody.LogSlots = params.MaxSlots()
			reqBody.Ctxt = input_b
			reqBody_m, _ := json.Marshal(reqBody)
			reqBody_b := bytes.NewBuffer(reqBody_m)
			respBody := DoReq(url, reqBody_b)
			resJson := Response{}
			json.Unmarshal(respBody, &resJson)
			result_b, err := base64.StdEncoding.DecodeString(resJson.Data)
			blind_auth.ReturnErr(err)
			ctxt := &rlwe.Ciphertext{}
			ctxt.UnmarshalBinary(result_b)
			resp := &Resp{}
			resp.Data, resp.Index = ctxt, i

			queue <- *resp
		}(u, index)
	}
	go func() {
		wg.Wait()
		close(queue)
	}()
	for v := range queue {
		results = append(results, v)
	}

	result := results[0].Data
	if len(results) > 1 {
		for i := 1; i < len(results); i++ {
			o.Eval.Add(result, results[i].Data, result)
		}
	}
	result_b, err := result.MarshalBinary()
	blind_auth.ReturnErr(err)
	return result_b
}
