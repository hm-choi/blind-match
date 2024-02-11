package modules

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
