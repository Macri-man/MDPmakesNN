package nnlib

import (
	"encoding/json"
	"os"
	"strings"
)

type serialLayer struct {
	Weights    [][]float64 `json:"weights"`
	Biases     []float64   `json:"biases"`
	Activation string      `json:"activation"`
}

type serialModel struct {
	Layers []serialLayer `json:"layers"`
}

// Save model to JSON file
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

// Load model from JSON file
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
