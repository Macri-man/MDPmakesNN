package main

import (
	"MDPmakesNN/mdplib"
	nn "MDPmakesNN/nnlib"
	"fmt"
)

func main() {
	model := nn.NewNeuralNetwork(
		[]int{2, 4, 2},
		[]nn.ActivationFunc{nn.ReLU{}, &nn.Softmax{}},
	)

	inputs := [][]float64{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
	}
	targets := [][]float64{
		{1, 0}, {0, 1}, {0, 1}, {1, 0},
	}

	for epoch := 0; epoch < 1000; epoch++ {
		model.TrainBatch(inputs, targets, 0.1)
	}

	preds := make([][]float64, len(inputs))
	for i, input := range inputs {
		preds[i] = model.Predict(input)
	}

	fmt.Printf("Accuracy: %.2f%%\n", nn.Accuracy(preds, targets)*100)

	err := model.Save("model.json")
	if err != nil {
		fmt.Println("Error saving model:", err)
	}

	mdp := mdplib.NewMDP([]mdplib.State{}, 0.9)

	// Load either JSON or CSV
	if err := mdp.LoadFromCSV("example/mdp.csv"); err != nil {
		panic(err)
	}
	// Or: mdp.LoadFromJSON("example/mdp.json")

	// Run policy iteration
	mdp.PolicyIteration()

	// Output results
	fmt.Println("Optimal Value Function:")
	for s, v := range mdp.ValueFunc {
		fmt.Printf("V(%s) = %.2f\n", s, v)
	}

	fmt.Println("\nOptimal Policy:")
	for s, a := range mdp.Policy {
		fmt.Printf("π(%s) = %s\n", s, a)
	}
}
