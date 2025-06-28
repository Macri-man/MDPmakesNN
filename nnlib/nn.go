package nnlib

import (
	"fmt"
)

// NeuralNetwork holds layers of the model
type NeuralNetwork struct {
	Layers []*Layer
}

// NewNeuralNetwork creates a NN from layer sizes and activations
func NewNeuralNetwork(sizes []int, activations []ActivationFunc) *NeuralNetwork {
	if len(sizes)-1 != len(activations) {
		panic("Number of activations must be one less than number of layers")
	}
	nn := &NeuralNetwork{}
	for i := 0; i < len(sizes)-1; i++ {
		nn.Layers = append(nn.Layers, NewLayer(sizes[i], sizes[i+1], activations[i]))
	}
	return nn
}

// Forward propagates input through all layers
func (nn *NeuralNetwork) Forward(input []float64) []float64 {
	for _, layer := range nn.Layers {
		input = layer.Forward(input)
	}
	return input
}

// Train on one example with cross-entropy loss by default
func (nn *NeuralNetwork) Train(input, target []float64, learningRate float64) {
	output := nn.Forward(input)
	_, grad := CrossEntropyLoss(output, target)
	errorGrad := grad

	for i := len(nn.Layers) - 1; i >= 0; i-- {
		errorGrad = nn.Layers[i].Backward(errorGrad, learningRate)
	}
}

// TrainBatch processes batch of samples, averages gradients
func (nn *NeuralNetwork) TrainBatch(inputs, targets [][]float64, learningRate float64) {
	batchSize := len(inputs)

	layerGrads := make([][][]float64, len(nn.Layers))
	layerBiasGrads := make([][]float64, len(nn.Layers))
	for i, layer := range nn.Layers {
		layerGrads[i] = make([][]float64, len(layer.Weights))
		for j := range layerGrads[i] {
			layerGrads[i][j] = make([]float64, len(layer.Weights[j]))
		}
		layerBiasGrads[i] = make([]float64, len(layer.Biases))
	}

	for idx := 0; idx < batchSize; idx++ {
		output := nn.Forward(inputs[idx])
		_, grad := CrossEntropyLoss(output, targets[idx])
		errorGrad := grad

		for l := len(nn.Layers) - 1; l >= 0; l-- {
			layer := nn.Layers[l]
			layer.Backward(errorGrad, 0) // no weight update yet
			errorGrad = make([]float64, len(layer.inputs))

			for j := range layer.inputs {
				for k := range layer.deltas {
					errorGrad[j] += layer.deltas[k] * layer.Weights[k][j]
					layerGrads[l][k][j] += layer.deltas[k] * layer.inputs[j]
				}
			}
			for k := range layer.deltas {
				layerBiasGrads[l][k] += layer.deltas[k]
			}
		}
	}

	for i, layer := range nn.Layers {
		for j := range layer.Weights {
			for k := range layer.Weights[j] {
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

// PrintWeights for debug
func (nn *NeuralNetwork) PrintWeights() {
	for i, layer := range nn.Layers {
		fmt.Printf("Layer %d weights:\n", i)
		for _, w := range layer.Weights {
			fmt.Println(w)
		}
		fmt.Printf("Biases: %v\n", layer.Biases)
	}
}
