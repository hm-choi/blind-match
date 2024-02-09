package api

import (
	"encoding/base64"
	"encoding/json"
	"net/http"

	"github.com/gin-gonic/gin"
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
