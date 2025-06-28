package nn

import "math"

// CrossEntropyLoss computes loss and gradient for classification
func CrossEntropyLoss(predicted, target []float64) (loss float64, grad []float64) {
	grad = make([]float64, len(predicted))
	for i := range predicted {
		p := math.Max(predicted[i], 1e-15)
		t := target[i]
		loss -= t * math.Log(p)
		grad[i] = predicted[i] - t
	}
	return
}

// MSELoss computes mean squared error loss and gradient (optional)
func MSELoss(predicted, target []float64) (loss float64, grad []float64) {
	grad = make([]float64, len(predicted))
	for i := range predicted {
		diff := predicted[i] - target[i]
		loss += diff * diff
		grad[i] = 2 * diff
	}
	loss /= float64(len(predicted))
	return
}
