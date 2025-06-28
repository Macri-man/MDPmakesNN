package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

// Activation functions
type ActivationFunc struct {
	Activate func(float64) float64
	Deriv    func(float64) float64
}

var ReLU = ActivationFunc{
	Activate: func(x float64) float64 {
		if x > 0 {
			return x
		}
		return 0
	},
	Deriv: func(x float64) float64 {
		if x > 0 {
			return 1
		}
		return 0
	},
}

var Tanh = ActivationFunc{
	Activate: math.Tanh,
	Deriv: func(x float64) float64 {
		t := math.Tanh(x)
		return 1 - t*t
	},
}

// NeuralNetwork struct
type NeuralNetwork struct {
	layerSizes     []int
	weights        [][][]float64
	biases         [][]float64
	learningRate   float64
	activationFunc ActivationFunc
}

// NewNeuralNetwork constructor
func NewNeuralNetwork(layerSizes []int, activation ActivationFunc) *NeuralNetwork {
	rand.Seed(time.Now().UnixNano())

	nn := &NeuralNetwork{
		layerSizes:     layerSizes,
		learningRate:   0.1,
		activationFunc: activation,
	}

	for i := 0; i < len(layerSizes)-1; i++ {
		nn.weights = append(nn.weights, randomMatrix(layerSizes[i+1], layerSizes[i]))
		nn.biases = append(nn.biases, randomArray(layerSizes[i+1]))
	}

	return nn
}

// Forward pass
func (nn *NeuralNetwork) Predict(input []float64) []float64 {
	a := input
	for i := 0; i < len(nn.weights); i++ {
		z := add(dot(nn.weights[i], a), nn.biases[i])
		a = apply(z, nn.activationFunc.Activate)
	}
	return a
}

// Train one example with backpropagation
func (nn *NeuralNetwork) Train(input, target []float64) {
	activations := [][]float64{input}
	zs := [][]float64{}

	// Forward
	for i := 0; i < len(nn.weights); i++ {
		z := add(dot(nn.weights[i], activations[i]), nn.biases[i])
		zs = append(zs, z)
		activations = append(activations, apply(z, nn.activationFunc.Activate))
	}

	// Backward
	delta := hadamard(
		subtract(target, activations[len(activations)-1]),
		apply(zs[len(zs)-1], nn.activationFunc.Deriv),
	)

	for i := len(nn.weights) - 1; i >= 0; i-- {
		aPrev := activations[i]
		dw := outer(delta, aPrev)

		addToMatrixScaled(nn.weights[i], dw, -nn.learningRate)
		addToArrayScaled(nn.biases[i], delta, -nn.learningRate)

		if i > 0 {
			delta = hadamard(
				dotT(nn.weights[i], delta),
				apply(zs[i-1], nn.activationFunc.Deriv),
			)
		}
	}
}

// Compute gradients without applying (for gradient checking)
func (nn *NeuralNetwork) ComputeGradients(input, target []float64) ([][][]float64, [][]float64) {
	activations := [][]float64{input}
	zs := [][]float64{}

	for i := 0; i < len(nn.weights); i++ {
		z := add(dot(nn.weights[i], activations[i]), nn.biases[i])
		zs = append(zs, z)
		activations = append(activations, apply(z, nn.activationFunc.Activate))
	}

	delta := hadamard(
		subtract(activations[len(activations)-1], target),
		apply(zs[len(zs)-1], nn.activationFunc.Deriv),
	)

	gradWeights := make([][][]float64, len(nn.weights))
	gradBiases := make([][]float64, len(nn.biases))

	for i := len(nn.weights) - 1; i >= 0; i-- {
		aPrev := activations[i]
		gradWeights[i] = outer(delta, aPrev)
		gradBiases[i] = delta

		if i > 0 {
			delta = hadamard(
				dotT(nn.weights[i], delta),
				apply(zs[i-1], nn.activationFunc.Deriv),
			)
		}
	}

	return gradWeights, gradBiases
}

func hasNaN(data interface{}) bool {
	switch v := data.(type) {
	case float64:
		return math.IsNaN(v)
	case []float64:
		for _, x := range v {
			if math.IsNaN(x) {
				return true
			}
		}
	case [][]float64:
		for _, arr := range v {
			if hasNaN(arr) {
				return true
			}
		}
	case [][][]float64:
		for _, mat := range v {
			if hasNaN(mat) {
				return true
			}
		}
	}
	return false
}

// Mean Squared Error Loss
func (nn *NeuralNetwork) Loss(output, target []float64) float64 {
	sum := 0.0
	for i := range output {
		diff := target[i] - output[i]
		sum += diff * diff
	}
	return sum / float64(len(output))
}

// Gradient check
func (nn *NeuralNetwork) GradientCheck(input, target []float64, epsilon float64) {
	fmt.Println("Starting gradient check...")
	gradW, gradB := nn.ComputeGradients(input, target)

	for l := range nn.weights {
		for i := range nn.weights[l] {
			for j := range nn.weights[l][i] {
				original := nn.weights[l][i][j]

				nn.weights[l][i][j] = original + epsilon
				lossPlus := nn.Loss(nn.Predict(input), target)

				nn.weights[l][i][j] = original - epsilon
				lossMinus := nn.Loss(nn.Predict(input), target)

				nn.weights[l][i][j] = original

				gradApprox := (lossPlus - lossMinus) / (2 * epsilon)
				gradBackprop := gradW[l][i][j]

				diff := math.Abs(gradApprox - gradBackprop)
				fmt.Printf("Layer %d Weight[%d][%d]: approx=%.6f backprop=%.6f diff=%.6f\n",
					l, i, j, gradApprox, gradBackprop, diff)
			}
		}
	}

	// Similarly for biases
	for l := range nn.biases {
		for i := range nn.biases[l] {
			original := nn.biases[l][i]

			nn.biases[l][i] = original + epsilon
			lossPlus := nn.Loss(nn.Predict(input), target)

			nn.biases[l][i] = original - epsilon
			lossMinus := nn.Loss(nn.Predict(input), target)

			nn.biases[l][i] = original

			gradApprox := (lossPlus - lossMinus) / (2 * epsilon)
			gradBackprop := gradB[l][i]

			diff := math.Abs(gradApprox - gradBackprop)
			fmt.Printf("Layer %d Bias[%d]: approx=%.6f backprop=%.6f diff=%.6f\n",
				l, i, gradApprox, gradBackprop, diff)
		}
	}
}

// Save model to JSON
func (nn *NeuralNetwork) SaveModel(filename string) error {
	data := map[string]interface{}{
		"layerSizes":     nn.layerSizes,
		"weights":        nn.weights,
		"biases":         nn.biases,
		"learningRate":   nn.learningRate,
		"activationFunc": "ReLU", // change to "Tanh" if using tanh
	}

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(data)
}

// Load model from JSON
func LoadModel(filename string) (*NeuralNetwork, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	data := map[string]interface{}{}
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&data); err != nil {
		return nil, err
	}

	layerSizes := []int{}
	for _, v := range data["layerSizes"].([]interface{}) {
		layerSizes = append(layerSizes, int(v.(float64)))
	}

	weights := parse3DFloat64(data["weights"].([]interface{}))
	biases := parse2DFloat64(data["biases"].([]interface{}))

	actName := data["activationFunc"].(string)
	var activation ActivationFunc
	if actName == "ReLU" {
		activation = ReLU
	} else {
		activation = Tanh
	}

	nn := &NeuralNetwork{
		layerSizes:     layerSizes,
		weights:        weights,
		biases:         biases,
		learningRate:   data["learningRate"].(float64),
		activationFunc: activation,
	}

	return nn, nil
}

func parse3DFloat64(data []interface{}) [][][]float64 {
	out := make([][][]float64, len(data))
	for i, layer := range data {
		layerArr := layer.([]interface{})
		out[i] = make([][]float64, len(layerArr))
		for j, row := range layerArr {
			rowArr := row.([]interface{})
			out[i][j] = make([]float64, len(rowArr))
			for k, val := range rowArr {
				out[i][j][k] = val.(float64)
			}
		}
	}
	return out
}

func parse2DFloat64(data []interface{}) [][]float64 {
	out := make([][]float64, len(data))
	for i, arr := range data {
		arrF := arr.([]interface{})
		out[i] = make([]float64, len(arrF))
		for j, val := range arrF {
			out[i][j] = val.(float64)
		}
	}
	return out
}

// ======= Helper math funcs ========

func randomMatrix(rows, cols int) [][]float64 {
	m := make([][]float64, rows)
	for i := range m {
		m[i] = make([]float64, cols)
		for j := range m[i] {
			m[i][j] = rand.Float64()*2 - 1
		}
	}
	return m
}

func randomArray(size int) []float64 {
	a := make([]float64, size)
	for i := range a {
		a[i] = rand.Float64()*2 - 1
	}
	return a
}

func dot(weights [][]float64, input []float64) []float64 {
	output := make([]float64, len(weights))
	for i := range weights {
		for j := range input {
			output[i] += weights[i][j] * input[j]
		}
	}
	return output
}

func dotT(weights [][]float64, delta []float64) []float64 {
	output := make([]float64, len(weights[0]))
	for i := range weights {
		for j := range weights[i] {
			output[j] += weights[i][j] * delta[i]
		}
	}
	return output
}

func apply(vec []float64, fn func(float64) float64) []float64 {
	out := make([]float64, len(vec))
	for i, v := range vec {
		out[i] = fn(v)
	}
	return out
}

func subtract(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] - b[i]
	}
	return out
}

func add(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

func hadamard(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] * b[i]
	}
	return out
}

func outer(a, b []float64) [][]float64 {
	m := make([][]float64, len(a))
	for i := range a {
		m[i] = make([]float64, len(b))
		for j := range b {
			m[i][j] = a[i] * b[j]
		}
	}
	return m
}

func addToMatrixScaled(matrix, delta [][]float64, scale float64) {
	for i := range matrix {
		for j := range matrix[i] {
			matrix[i][j] += delta[i][j] * scale
		}
	}
}

func addToArrayScaled(arr, delta []float64, scale float64) {
	for i := range arr {
		arr[i] += delta[i] * scale
	}
}

/*
// ======= MAIN - XOR example ========

func main() {
	// Use ReLU or Tanh activation here:
	nn := NewNeuralNetwork([]int{2, 4, 4, 1}, ReLU)

	// XOR dataset
	data := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	// Train for 10,000 epochs
	for i := 0; i < 10000; i++ {
		for j := range data {
			nn.Train(data[j], targets[j])
		}
	}

	// Test predictions
	fmt.Println("Predictions after training:")
	for _, input := range data {
		out := nn.Predict(input)
		fmt.Printf("Input: %v Output: %.4f\n", input, out[0])
	}

	// Gradient check on first training example
	nn.GradientCheck(data[0], targets[0], 1e-4)

	// Save model
	err := nn.SaveModel("model.json")
	if err != nil {
		fmt.Println("Error saving model:", err)
	} else {
		fmt.Println("Model saved to model.json")
	}

	// Load model
	nn2, err := LoadModel("model.json")
	if err != nil {
		fmt.Println("Error loading model:", err)
	} else {
		fmt.Println("Model loaded from model.json")
		// Test loaded model
		for _, input := range data {
			out := nn2.Predict(input)
			fmt.Printf("Loaded model prediction: Input %v Output %.4f\n", input, out[0])
		}
	}
}

*/
