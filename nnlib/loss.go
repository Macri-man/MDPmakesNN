package nnlib

import "math"

// CrossEntropyLoss computes the cross-entropy loss and its gradient.
// predicted: output probabilities (after softmax), target: one-hot encoded labels.
func CrossEntropyLoss(predicted, target []float64) (loss float64, grad []float64) {
	const epsilon = 1e-15
	grad = make([]float64, len(predicted))

	for i := range predicted {
		// Clamp predicted probabilities for numerical stability
		p := math.Min(math.Max(predicted[i], epsilon), 1-epsilon)
		t := target[i]

		// Cross-entropy loss component for class i
		loss -= t * math.Log(p)

		// Gradient of cross-entropy wrt predicted (assuming softmax output)
		grad[i] = p - t
	}
	return loss, grad
}

// MSELoss computes mean squared error loss and its gradient.
// predicted: predicted values, target: true values.
func MSELoss(predicted, target []float64) (loss float64, grad []float64) {
	grad = make([]float64, len(predicted))
	for i := range predicted {
		diff := predicted[i] - target[i]
		loss += diff * diff
		grad[i] = 2 * diff
	}
	loss /= float64(len(predicted))
	return loss, grad
}
