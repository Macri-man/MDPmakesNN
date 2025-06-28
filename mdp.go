package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"os"
	"strconv"
	"strings"
)

type State string
type Action string

type Transition struct {
	NextState State   `json:"next"`
	Prob      float64 `json:"prob"`
	Reward    float64 `json:"reward"`
}

type RawTransition struct {
	State  State   `json:"state"`
	Action Action  `json:"action"`
	Next   State   `json:"next"`
	Prob   float64 `json:"prob"`
	Reward float64 `json:"reward"`
}

type RawState struct {
	Name     State `json:"name"`
	Terminal bool  `json:"terminal"`
}

type MDPInput struct {
	Discount    float64         `json:"discount"`
	Iterations  int             `json:"iterations"`
	Epsilon     float64         `json:"epsilon"`
	States      []RawState      `json:"states"`
	Transitions []RawTransition `json:"transitions"`
}

type MDP struct {
	States      []State
	Terminal    map[State]bool
	Actions     map[State][]Action
	Transitions map[State]map[Action][]Transition
	Discount    float64
	Value       map[State]float64
	Policy      map[State]Action
	Iterations  int
	Epsilon     float64
}

// Load MDP from CSV
func LoadMDPFromCSV(filename string) (*MDP, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Read() // skip header

	mdp := &MDP{
		Terminal:    make(map[State]bool),
		Actions:     make(map[State][]Action),
		Transitions: make(map[State]map[Action][]Transition),
		Value:       make(map[State]float64),
		Policy:      make(map[State]Action),
		Discount:    0.9,
		Iterations:  100,
		Epsilon:     0.01,
	}

	seenStates := make(map[State]bool)

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		}

		state := State(record[0])
		action := Action(record[1])
		next := State(record[2])
		prob, _ := strconv.ParseFloat(record[3], 64)
		reward, _ := strconv.ParseFloat(record[4], 64)
		terminal := strings.ToLower(record[5]) == "true"

		if !seenStates[state] {
			mdp.States = append(mdp.States, state)
			seenStates[state] = true
		}

		if terminal {
			mdp.Terminal[state] = true
			continue
		}

		if _, ok := mdp.Transitions[state]; !ok {
			mdp.Transitions[state] = make(map[Action][]Transition)
		}
		mdp.Transitions[state][action] = append(
			mdp.Transitions[state][action],
			Transition{NextState: next, Prob: prob, Reward: reward},
		)

		if !contains(mdp.Actions[state], action) && action != "" {
			mdp.Actions[state] = append(mdp.Actions[state], action)
		}
	}

	return mdp, nil
}

func LoadMDPFromJSON(filename string) (*MDP, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var input MDPInput
	if err := json.Unmarshal(data, &input); err != nil {
		return nil, err
	}

	mdp := &MDP{
		Terminal:    make(map[State]bool),
		Actions:     make(map[State][]Action),
		Transitions: make(map[State]map[Action][]Transition),
		Value:       make(map[State]float64),
		Policy:      make(map[State]Action),
		Discount:    input.Discount,
		Iterations:  input.Iterations,
		Epsilon:     input.Epsilon,
	}

	for _, s := range input.States {
		mdp.States = append(mdp.States, s.Name)
		mdp.Terminal[s.Name] = s.Terminal
	}

	for _, rt := range input.Transitions {
		if _, ok := mdp.Transitions[rt.State]; !ok {
			mdp.Transitions[rt.State] = make(map[Action][]Transition)
		}
		mdp.Transitions[rt.State][rt.Action] = append(
			mdp.Transitions[rt.State][rt.Action],
			Transition{NextState: rt.Next, Prob: rt.Prob, Reward: rt.Reward},
		)
		if !contains(mdp.Actions[rt.State], rt.Action) {
			mdp.Actions[rt.State] = append(mdp.Actions[rt.State], rt.Action)
		}
	}

	return mdp, nil
}

func LoadMDP(filename string) (*MDP, error) {
	switch {
	case strings.HasSuffix(filename, ".json"):
		return LoadMDPFromJSON(filename)
	case strings.HasSuffix(filename, ".csv"):
		return LoadMDPFromCSV(filename)
	default:
		return nil, fmt.Errorf("unsupported file format: %s", filename)
	}
}

func contains[T comparable](slice []T, val T) bool {
	for _, x := range slice {
		if x == val {
			return true
		}
	}
	return false
}

func (mdp *MDP) ValueIteration() {
	for i := 0; i < mdp.Iterations; i++ {
		delta := 0.0
		newValue := make(map[State]float64)

		for _, s := range mdp.States {
			if mdp.Terminal[s] {
				newValue[s] = 0 // Or fixed value if desired
				continue
			}

			maxValue := math.Inf(-1)
			var bestAction Action

			for _, a := range mdp.Actions[s] {
				expectedValue := 0.0
				for _, t := range mdp.Transitions[s][a] {
					expectedValue += t.Prob * (t.Reward + mdp.Discount*mdp.Value[t.NextState])
				}
				if expectedValue > maxValue {
					maxValue = expectedValue
					bestAction = a
				}
			}

			newValue[s] = maxValue
			mdp.Policy[s] = bestAction
			delta = math.Max(delta, math.Abs(mdp.Value[s]-newValue[s]))
		}

		mdp.Value = newValue

		if delta < mdp.Epsilon {
			break
		}
	}
}

/*
func main() {
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
*/
