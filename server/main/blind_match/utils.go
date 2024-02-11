package blind_match

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"

	"github.com/nfnt/resize"
	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

const (
	hSize, wSize  = 112, 112
	blockSize     = 32
	gridHeight    = 13
	gridWidth     = 13
	boxesPerCell  = 5
	numClasses    = 20
	envConfPrefix = "yolo"
)

func Example_gorgonia() {
	backend := gorgonnx.NewGraph()
	_ = onnx.NewModel(backend)
	// read the onnx model
	var f io.Reader
	var err error
	f, err = os.Open("../../dataset/AJ_Cook_0001.jpg")
	if err != nil {
		log.Fatal(err)
	}
	defer f.(*os.File).Close()
	img, err := jpeg.Decode(f)
	if err != nil {
		log.Fatal(err)
	}
	// find the resize scale
	imgRescaled := image.NewNRGBA(image.Rect(0, 0, wSize, hSize))
	color := color.RGBA{0, 0, 0, 255}

	draw.Draw(imgRescaled, imgRescaled.Bounds(), &image.Uniform{color}, image.ZP, draw.Src)
	var m image.Image
	if (img.Bounds().Max.X - img.Bounds().Min.X) > (img.Bounds().Max.Y - img.Bounds().Min.Y) {
		// scaleFactor := float32(img.Bounds().Max.Y-img.Bounds().Min.Y) / float32(hSize)
		m = resize.Resize(0, hSize, img, resize.Lanczos3)
	} else {
		// scaleFactor := float32(img.Bounds().Max.X-img.Bounds().Min.X) / float32(wSize)
		m = resize.Resize(wSize, 0, img, resize.Lanczos3)
	}
	switch m.(type) {
	case *image.NRGBA:
		draw.Draw(imgRescaled, imgRescaled.Bounds(), m.(*image.NRGBA), image.ZP, draw.Src)
	case *image.YCbCr:
		draw.Draw(imgRescaled, imgRescaled.Bounds(), m.(*image.YCbCr), image.ZP, draw.Src)
	default:
		log.Fatal("unhandled type")
	}

	// inputT := tensor.New(tensor.WithShape(1, 3, hSize, wSize), tensor.Of(tensor.Float32))
	//err = images.ImageToBCHW(img, inputT)
	// err = images.ImageToBCHW(imgRescaled, inputT)
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// Decode it into the model
	// inputT := tensor.New(tensor.WithShape(1, 3, 112, 112), tensor.Of(tensor.Float32))
	// inputT.
	// model.SetInput(0, inputT)

}

func ReturnErr(err error) {
	if err != nil {
		panic(err)
	}
}
func GenerateVector(N int) ([]float64, []float64) {
	result_vec1, result_vec2 := make([]float64, N), make([]float64, N)
	for i := 0; i < N; i++ {
		result_vec1[i] = 2*rand.Float64() - 1
		result_vec2[i] = 2*rand.Float64() - 1
	}
	return result_vec1, result_vec2
}

func ChangeVec(ctxts1 []float64, ctxts2 []float64, interval_size int, N int) ([][]float64, [][]float64) {
	changed1, changed2 := make([][]float64, N), make([][]float64, N)
	if len(changed1) != len(changed2) {
		panic("Error! Size of two chipertexts are not same!")
	}
	// Total 32,768
	// interval_size = 64
	// 32,768 = 512 * N * (64/N) = (Total/interval_size) * N * (interval_size/N)
	total := 32768
	interval_size_ := interval_size / N

	for t := 0; t < total/interval_size_; t++ {
		for n := 0; n < N; n++ {
			for i := 0; i < interval_size_; i++ {
				changed1[n] = append(changed1[n], ctxts1[N*t+total/interval_size_*n+i])
				changed2[n] = append(changed2[n], ctxts2[N*t*n+i])
			}
		}
	}
	return changed1, changed2
}

func ReadCSV(file_name string) []float64 {
	file, _ := os.Open(file_name)
	rdr := csv.NewReader(bufio.NewReader(file))
	rows, _ := rdr.ReadAll()
	result_vec := []float64{}
	for i, row := range rows {
		for j := range row {
			f, _ := strconv.ParseFloat(rows[i][j], 64)
			result_vec = append(result_vec, f)
		}
	}
	return result_vec
}
func CreateCSV() {
	N := 512 * 5000
	vec := make([]string, N)

	for i := 0; i < N; i++ {
		vec[i] = fmt.Sprintf("%f", 2*rand.Float64()-1)
	}
	// 파일 생성
	file, err := os.Create("./output.csv")
	if err != nil {
		panic(err)
	}
	// csv writer 생성
	wr := csv.NewWriter(bufio.NewWriter(file))
	// csv 내용 쓰기
	wr.Write(vec)
	wr.Flush()
}

func Cossim(vec1 []float64, vec2 []float64, interval_size int) []float64 {
	if len(vec1) != len(vec2) {
		panic("[Error] Two input vector size are different")
	}
	N := len(vec1) / interval_size
	result_slice := make([]float64, N)

	for i := 0; i < N; i++ {
		var result float64 = 0
		for j := 0; j < interval_size; j++ {
			result += vec1[interval_size*i+j] * vec2[interval_size*i+j]
		}
		result_slice[i] = result
	}
	return result_slice
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

func PowerOfTwo(n int) int {
	result := 1
	for i := 0; i < n; i++ {
		result *= 2
	}
	return result
}

func Normalization(input_vec []float64, N int) []float64 {
	result := make([]float64, len(input_vec))
	for i := 0; i < len(input_vec)/N; i++ {
		sum := 0.0
		for j := 0; j < N; j++ {
			sum += input_vec[i*N+j] * input_vec[i*N+j]
		}
		for j := 0; j < N; j++ {
			result[i*N+j] = input_vec[i*N+j] / math.Sqrt(sum)
		}
	}
	return result
}
