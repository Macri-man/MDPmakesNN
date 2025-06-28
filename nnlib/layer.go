package nnlib

import (
	"math/rand"
	"time"
)

// Layer represents a fully connected NN layer
type Layer struct {
	Weights    [][]float64
	Biases     []float64
	Activation ActivationFunc

	inputs  []float64
	outputs []float64
	deltas  []float64
}

// NewLayer initializes a new fully connected layer
func NewLayer(inputSize, outputSize int, activation ActivationFunc) *Layer {
	rand.Seed(time.Now().UnixNano())
	w := make([][]float64, outputSize)
	for i := range w {
		w[i] = make([]float64, inputSize)
		for j := range w[i] {
			w[i][j] = rand.Float64()*0.2 - 0.1
		}
	}
	b := make([]float64, outputSize)
	return &Layer{
		Weights:    w,
		Biases:     b,
		Activation: activation,
	}
}

// Forward propagates input through layer
func (l *Layer) Forward(input []float64) []float64 {
	l.inputs = input
	output := make([]float64, len(l.Weights))

	for i := range l.Weights {
		sum := l.Biases[i]
		for j := range input {
			sum += l.Weights[i][j] * input[j]
		}
		output[i] = l.Activation.Activate(sum)
	}

	// Special case for Softmax activation applied to entire output vector
	if softmax, ok := l.Activation.(*Softmax); ok {
		output = softmax.ActivateVector(output)
	}

	l.outputs = output
	return output
}

// Backward propagates error, updates weights if learningRate > 0
func (l *Layer) Backward(errorGrad []float64, learningRate float64) []float64 {
	l.deltas = make([]float64, len(l.outputs))

	// For softmax + cross-entropy, derivative simplified
	if _, ok := l.Activation.(*Softmax); ok {
		for i := range l.outputs {
			l.deltas[i] = errorGrad[i]
		}
	} else {
		for i := range l.outputs {
			l.deltas[i] = errorGrad[i] * l.Activation.Derivative(l.outputs[i])
		}
	}

	prevError := make([]float64, len(l.inputs))
	for j := range l.inputs {
		sum := 0.0
		for i := range l.deltas {
			sum += l.deltas[i] * l.Weights[i][j]
		}
		prevError[j] = sum
	}

	if learningRate > 0 {
		for i := range l.Weights {
			for j := range l.Weights[i] {
				l.Weights[i][j] -= learningRate * l.deltas[i] * l.inputs[j]
			}
			l.Biases[i] -= learningRate * l.deltas[i]
		}
	}

	return prevError
}
