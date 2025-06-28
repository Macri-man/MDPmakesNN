package nn

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"
)

// Activation function interface
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
		exp := math.Exp(v - maxVal) // for numerical stability
		output[i] = exp
		expSum += exp
	}
	for i := range output {
		output[i] /= expSum
	}
	s.lastOutput = output
	return output
}

// Not used in most practical backprop, but placeholder to satisfy interface
func (s *Softmax) Activate(x float64) float64 {
	return x // Softmax works on vector, not scalar
}

// Derivative for softmax (used only with cross-entropy loss)
func (s *Softmax) Derivative(x float64) float64 {
	return 1 // placeholder, not used directly
}

// Layer represents a fully connected layer
type Layer struct {
	Weights    [][]float64
	Biases     []float64
	Activation ActivationFunc

	inputs  []float64
	outputs []float64
	deltas  []float64
}

// NewLayer creates a new fully connected layer
func NewLayer(inputSize, outputSize int, activation ActivationFunc) *Layer {
	rand.Seed(time.Now().UnixNano())
	w := make([][]float64, outputSize)
	for i := range w {
		w[i] = make([]float64, inputSize)
		for j := range w[i] {
			// Initialize small random weights
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

// Forward propagates input through the layer
func (l *Layer) Forward(input []float64) []float64 {
	l.inputs = input
	output := make([]float64, len(l.Weights))
	for i := 0; i < len(l.Weights); i++ {
		sum := l.Biases[i]
		for j := 0; j < len(input); j++ {
			sum += l.Weights[i][j] * input[j]
		}
		output[i] = l.Activation.Activate(sum)
	}
	l.outputs = output
	return output
}

// Backward calculates error and updates weights and biases
func (l *Layer) Backward(errorGrad []float64, learningRate float64) []float64 {
	l.deltas = make([]float64, len(l.outputs))
	for i := range l.outputs {
		l.deltas[i] = errorGrad[i] * l.Activation.Derivative(l.outputs[i])
	}

	// Calculate error gradient for previous layer
	prevError := make([]float64, len(l.inputs))
	for j := 0; j < len(l.inputs); j++ {
		sum := 0.0
		for i := 0; i < len(l.deltas); i++ {
			sum += l.deltas[i] * l.Weights[i][j]
		}
		prevError[j] = sum
	}

	// Update weights and biases
	for i := 0; i < len(l.Weights); i++ {
		for j := 0; j < len(l.Weights[i]); j++ {
			l.Weights[i][j] -= learningRate * l.deltas[i] * l.inputs[j]
		}
		l.Biases[i] -= learningRate * l.deltas[i]
	}

	return prevError
}

// NeuralNetwork represents a simple feed-forward NN
type NeuralNetwork struct {
	Layers []*Layer
}

// NewNeuralNetwork creates a network with given layer sizes and activations
func NewNeuralNetwork(sizes []int, activations []ActivationFunc) *NeuralNetwork {
	if len(sizes)-1 != len(activations) {
		panic("Number of activations must be one less than number of layers")
	}
	nn := &NeuralNetwork{}
	for i := 0; i < len(sizes)-1; i++ {
		layer := NewLayer(sizes[i], sizes[i+1], activations[i])
		nn.Layers = append(nn.Layers, layer)
	}
	return nn
}

// Forward forward propagates input through all layers
func (nn *NeuralNetwork) Forward(input []float64) []float64 {
	for _, layer := range nn.Layers {
		input = layer.Forward(input)
	}
	return input
}

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

// Train trains the network on one example with mean squared error loss
func (nn *NeuralNetwork) Train(input, target []float64, learningRate float64) {
	output := nn.Forward(input)

	// Calculate output error gradient (dLoss/dOutput)
	errorGrad := make([]float64, len(output))
	for i := range output {
		errorGrad[i] = 2 * (output[i] - target[i])
	}

	// Backpropagate through layers in reverse order
	for i := len(nn.Layers) - 1; i >= 0; i-- {
		errorGrad = nn.Layers[i].Backward(errorGrad, learningRate)
	}
}

// TrainBatch trains on a batch of input/target pairs
func (nn *NeuralNetwork) TrainBatch(inputs, targets [][]float64, learningRate float64) {
	batchSize := len(inputs)

	// Accumulate gradients
	layerGrads := make([][][]float64, len(nn.Layers))
	layerBiasGrads := make([][]float64, len(nn.Layers))
	for i, layer := range nn.Layers {
		layerGrads[i] = make([][]float64, len(layer.Weights))
		for j := range layerGrads[i] {
			layerGrads[i][j] = make([]float64, len(layer.Weights[j]))
		}
		layerBiasGrads[i] = make([]float64, len(layer.Biases))
	}

	// Process each sample
	for i := 0; i < batchSize; i++ {
		output := nn.Forward(inputs[i])
		_, grad := CrossEntropyLoss(output, targets[i])

		// Backpropagate
		errorGrad := grad
		for l := len(nn.Layers) - 1; l >= 0; l-- {
			layer := nn.Layers[l]
			layer.Backward(errorGrad, 0) // Don't update weights yet
			errorGrad = make([]float64, len(layer.inputs))
			for j := 0; j < len(layer.inputs); j++ {
				for k := 0; k < len(layer.deltas); k++ {
					errorGrad[j] += layer.deltas[k] * layer.Weights[k][j]
					layerGrads[l][k][j] += layer.deltas[k] * layer.inputs[j]
				}
			}
			for k := 0; k < len(layer.deltas); k++ {
				layerBiasGrads[l][k] += layer.deltas[k]
			}
		}
	}

	// Apply averaged gradients
	for i, layer := range nn.Layers {
		for j := 0; j < len(layer.Weights); j++ {
			for k := 0; k < len(layer.Weights[j]); k++ {
				layer.Weights[j][k] -= learningRate * layerGrads[i][j][k] / float64(batchSize)
			}
			layer.Biases[j] -= learningRate * layerBiasGrads[i][j] / float64(batchSize)
		}
	}
}

// Predict runs forward pass only
func (nn *NeuralNetwork) Predict(input []float64) []float64 {
	return nn.Forward(input)
}

// Utility function to print weights (for debug)
func (nn *NeuralNetwork) PrintWeights() {
	for i, layer := range nn.Layers {
		fmt.Printf("Layer %d weights:\n", i)
		for _, w := range layer.Weights {
			fmt.Println(w)
		}
		fmt.Printf("Biases: %v\n", layer.Biases)
	}
}

func Load(filename string) (*NeuralNetwork, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var s serialModel
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, err
	}

	nn := &NeuralNetwork{}
	for _, l := range s.Layers {
		layer := &Layer{
			Weights:    l.Weights,
			Biases:     l.Biases,
			Activation: activationFromName(l.Activation),
		}
		nn.Layers = append(nn.Layers, layer)
	}
	return nn, nil
}

func (nn *NeuralNetwork) Save(filename string) error {
	s := serialModel{}
	for _, layer := range nn.Layers {
		s.Layers = append(s.Layers, serialLayer{
			Weights:    layer.Weights,
			Biases:     layer.Biases,
			Activation: activationName(layer.Activation),
		})
	}

	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filename, data, 0644)
}

// Activation name helpers
func activationName(act ActivationFunc) string {
	switch act.(type) {
	case Sigmoid:
		return "sigmoid"
	case ReLU:
		return "relu"
	case *Softmax:
		return "softmax"
	default:
		return "unknown"
	}
}

func activationFromName(name string) ActivationFunc {
	switch strings.ToLower(name) {
	case "sigmoid":
		return Sigmoid{}
	case "relu":
		return ReLU{}
	case "softmax":
		return &Softmax{}
	default:
		panic("unknown activation: " + name)
	}
}
