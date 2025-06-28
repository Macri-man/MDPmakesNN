package nnlib

import (
	"math"
)

// ActivationFunc defines interface for scalar activation functions.
type ActivationFunc interface {
	Activate(x float64) float64
	Derivative(x float64) float64
}

// VectorActivationFunc extends ActivationFunc for vector activations (optional).
type VectorActivationFunc interface {
	ActivationFunc
	ActivateVector(input []float64) []float64
}

// --------------------
// Sigmoid activation
// --------------------
type Sigmoid struct{}

func (s Sigmoid) Activate(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (s Sigmoid) Derivative(x float64) float64 {
	sig := s.Activate(x)
	return sig * (1 - sig)
}

// --------------------
// ReLU activation
// --------------------
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

// --------------------
// LeakyReLU activation (alpha = 0.01 by default)
// --------------------
type LeakyReLU struct {
	Alpha float64
}

func (l LeakyReLU) Activate(x float64) float64 {
	if x > 0 {
		return x
	}
	return l.Alpha * x
}

func (l LeakyReLU) Derivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return l.Alpha
}

// --------------------
// Tanh activation
// --------------------
type Tanh struct{}

func (t Tanh) Activate(x float64) float64 {
	return math.Tanh(x)
}

func (t Tanh) Derivative(x float64) float64 {
	val := math.Tanh(x)
	return 1 - val*val
}

// --------------------
// Linear activation (identity)
// --------------------
type Linear struct{}

func (l Linear) Activate(x float64) float64 {
	return x
}

func (l Linear) Derivative(x float64) float64 {
	return 1
}

// --------------------
// ELU activation (Exponential Linear Unit)
// --------------------
type ELU struct {
	Alpha float64
}

func (e ELU) Activate(x float64) float64 {
	if x >= 0 {
		return x
	}
	return e.Alpha * (math.Exp(x) - 1)
}

func (e ELU) Derivative(x float64) float64 {
	if x >= 0 {
		return 1
	}
	return e.Alpha * math.Exp(x)
}

// --------------------
// Swish activation (x * sigmoid(x))
// --------------------
type Swish struct{}

func (s Swish) Activate(x float64) float64 {
	return x / (1 + math.Exp(-x))
}

func (s Swish) Derivative(x float64) float64 {
	sig := 1 / (1 + math.Exp(-x))
	return sig + x*sig*(1-sig)
}

// --------------------
// Softmax activation (vector only)
// --------------------
type Softmax struct {
	lastOutput []float64
}

// ActivateVector applies softmax over input slice and returns probabilities
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
		exp := math.Exp(v - maxVal) // numerical stability trick
		output[i] = exp
		expSum += exp
	}

	for i := range output {
		output[i] /= expSum
	}

	s.lastOutput = output
	return output
}

// Softmax scalar activation is a no-op (softmax works on vectors)
func (s *Softmax) Activate(x float64) float64 {
	return x
}

// Derivative placeholder (usually handled with cross-entropy loss)
func (s *Softmax) Derivative(x float64) float64 {
	return 1
}

// --------------------
// Vector helper to apply scalar activation elementwise
// --------------------
func ApplyActivationVec(vec []float64, act ActivationFunc) []float64 {
	out := make([]float64, len(vec))
	for i, v := range vec {
		out[i] = act.Activate(v)
	}
	return out
}

func ApplyDerivativeVec(vec []float64, act ActivationFunc) []float64 {
	out := make([]float64, len(vec))
	for i, v := range vec {
		out[i] = act.Derivative(v)
	}
	return out
}
