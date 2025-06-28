package nn

// ArgMax returns index of max value in slice
func ArgMax(vec []float64) int {
	maxIdx := 0
	maxVal := vec[0]
	for i, v := range vec {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

// Accuracy computes classification accuracy
func Accuracy(predictions, targets [][]float64) float64 {
	correct := 0
	for i := range predictions {
		if ArgMax(predictions[i]) == ArgMax(targets[i]) {
			correct++
		}
	}
	return float64(correct) / float64(len(predictions))
}
