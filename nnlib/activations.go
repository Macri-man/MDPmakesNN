package nn

import (
	"math"
)

// ActivationFunc interface for activations
type ActivationFunc interface {
	Activate(x float64) float64
	Derivative(x float64) float64
}

// Sigmoid activation
type Sigmoid struct{}

func (s Sigmoid) Activate(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
func (s Sigmoid) Derivative(x float64) float64 {
	sig := s.Activate(x)
	return sig * (1 - sig)
}

// ReLU activation
type ReLU struct{}

func (r ReLU) Activate(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}
func (r ReLU) Derivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// Softmax activation (works on vectors)
type Softmax struct {
	lastOutput []float64
}

func (s *Softmax) ActivateVector(input []float64) []float64 {
	maxVal := input[0]
	for _, v := range input {
		if v > maxVal {
			maxVal = v
		}
	}
	expSum := 0.0
	output := make([]float64, len(input))
	for i, v := range input {
		exp := math.Exp(v - maxVal) // numerical stability
		output[i] = exp
		expSum += exp
	}
	for i := range output {
		output[i] /= expSum
	}
	s.lastOutput = output
	return output
}

func (s *Softmax) Activate(x float64) float64 {
	// Softmax works on vectors, not scalar; no-op here
	return x
}
func (s *Softmax) Derivative(x float64) float64 {
	// Usually combined with cross-entropy, so not used standalone
	return 1
}
