package main

import (
	"fmt"
	"os"
)

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

	mdp, err := LoadMDP("mdp.json")
	if err != nil {
		fmt.Println("Error:", err)
		os.Exit(1)
	}

	mdp.ValueIteration()

	fmt.Println("Optimal Value Function:")
	for _, s := range mdp.States {
		fmt.Printf("V(%s) = %.2f\n", s, mdp.Value[s])
	}

	fmt.Println("\nOptimal Policy:")
	for _, s := range mdp.States {
		if mdp.Terminal[s] {
			fmt.Printf("π(%s) = [terminal]\n", s)
		} else {
			fmt.Printf("π(%s) = %s\n", s, mdp.Policy[s])
		}
	}

	mdp, err = LoadMDPFromCSV("mdp.csv")
	if err != nil {
		fmt.Println("Error loading CSV:", err)
		return
	}

	mdp.ValueIteration()

	fmt.Println("Optimal Value Function:")
	for _, s := range mdp.States {
		fmt.Printf("V(%s) = %.2f\n", s, mdp.Value[s])
	}

	fmt.Println("\nOptimal Policy:")
	for _, s := range mdp.States {
		if mdp.Terminal[s] {
			fmt.Printf("π(%s) = [terminal]\n", s)
		} else {
			fmt.Printf("π(%s) = %s\n", s, mdp.Policy[s])
		}
	}
}
