package mdplib

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"strconv"
)

type State string
type Action string

type Transition struct {
	NextState State
	Prob      float64
	Reward    float64
}

type MDP struct {
	States        []State
	Actions       map[State][]Action
	Transitions   map[State]map[Action][]Transition
	Discount      float64
	ValueFunc     map[State]float64
	Policy        map[State]Action
	Tolerance     float64
	MaxIterations int
}

func NewMDP(states []State, discount float64) *MDP {
	return &MDP{
		States:        states,
		Actions:       make(map[State][]Action),
		Transitions:   make(map[State]map[Action][]Transition),
		Discount:      discount,
		ValueFunc:     make(map[State]float64),
		Policy:        make(map[State]Action),
		Tolerance:     1e-6,
		MaxIterations: 1000,
	}
}

func (m *MDP) AddAction(state State, action Action, transitions []Transition) {
	m.Actions[state] = append(m.Actions[state], action)
	if m.Transitions[state] == nil {
		m.Transitions[state] = make(map[Action][]Transition)
	}
	m.Transitions[state][action] = transitions
}

func (m *MDP) ValueIteration() {
	for i := 0; i < m.MaxIterations; i++ {
		delta := 0.0
		newValues := make(map[State]float64)
		for _, s := range m.States {
			bestValue := math.Inf(-1)
			for _, a := range m.Actions[s] {
				v := 0.0
				for _, t := range m.Transitions[s][a] {
					v += t.Prob * (t.Reward + m.Discount*m.ValueFunc[t.NextState])
				}
				if v > bestValue {
					bestValue = v
				}
			}
			newValues[s] = bestValue
			delta = math.Max(delta, math.Abs(bestValue-m.ValueFunc[s]))
		}
		m.ValueFunc = newValues
		if delta < m.Tolerance {
			break
		}
	}
}

func (m *MDP) ExtractPolicy() {
	for _, s := range m.States {
		bestAction := Action("")
		bestValue := math.Inf(-1)
		for _, a := range m.Actions[s] {
			v := 0.0
			for _, t := range m.Transitions[s][a] {
				v += t.Prob * (t.Reward + m.Discount*m.ValueFunc[t.NextState])
			}
			if v > bestValue {
				bestValue = v
				bestAction = a
			}
		}
		m.Policy[s] = bestAction
	}
}

func (m *MDP) PolicyIteration() {
	// Initialize arbitrary policy
	for _, s := range m.States {
		if len(m.Actions[s]) > 0 {
			m.Policy[s] = m.Actions[s][0]
		}
	}

	for i := 0; i < m.MaxIterations; i++ {
		m.policyEvaluation()
		policyStable := true

		for _, s := range m.States {
			oldAction := m.Policy[s]
			bestAction := oldAction
			bestValue := math.Inf(-1)

			for _, a := range m.Actions[s] {
				v := 0.0
				for _, t := range m.Transitions[s][a] {
					v += t.Prob * (t.Reward + m.Discount*m.ValueFunc[t.NextState])
				}
				if v > bestValue {
					bestValue = v
					bestAction = a
				}
			}

			m.Policy[s] = bestAction
			if bestAction != oldAction {
				policyStable = false
			}
		}

		if policyStable {
			break
		}
	}
}

func (m *MDP) policyEvaluation() {
	for iter := 0; iter < m.MaxIterations; iter++ {
		delta := 0.0
		newValues := make(map[State]float64)

		for _, s := range m.States {
			a := m.Policy[s]
			v := 0.0
			for _, t := range m.Transitions[s][a] {
				v += t.Prob * (t.Reward + m.Discount*m.ValueFunc[t.NextState])
			}
			newValues[s] = v
			delta = math.Max(delta, math.Abs(v-m.ValueFunc[s]))
		}

		m.ValueFunc = newValues
		if delta < m.Tolerance {
			break
		}
	}
}

type RawTransition struct {
	State     string  `json:"state"`
	Action    string  `json:"action"`
	NextState string  `json:"next_state"`
	Prob      float64 `json:"prob"`
	Reward    float64 `json:"reward"`
}

func (m *MDP) LoadFromCSV(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.Read() // skip header

	for {
		record, err := reader.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}

		s := State(record[0])
		a := Action(record[1])
		ns := State(record[2])
		p, _ := strconv.ParseFloat(record[3], 64)
		r, _ := strconv.ParseFloat(record[4], 64)

		m.States = appendIfMissing(m.States, s)
		m.States = appendIfMissing(m.States, ns)
		m.Actions[s] = appendIfMissingAction(m.Actions[s], a)
		m.Transitions[s][a] = append(m.Transitions[s][a], Transition{NextState: ns, Prob: p, Reward: r})
	}
	return nil
}

func (m *MDP) LoadFromJSON(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var raw []RawTransition
	err = json.Unmarshal(data, &raw)
	if err != nil {
		return err
	}

	for _, entry := range raw {
		s := State(entry.State)
		a := Action(entry.Action)
		ns := State(entry.NextState)

		m.States = appendIfMissing(m.States, s)
		m.States = appendIfMissing(m.States, ns)
		m.Actions[s] = appendIfMissingAction(m.Actions[s], a)
		m.Transitions[s][a] = append(m.Transitions[s][a], Transition{NextState: ns, Prob: entry.Prob, Reward: entry.Reward})
	}
	return nil
}

func appendIfMissing(slice []State, s State) []State {
	for _, v := range slice {
		if v == s {
			return slice
		}
	}
	return append(slice, s)
}

func appendIfMissingAction(slice []Action, a Action) []Action {
	for _, v := range slice {
		if v == a {
			return slice
		}
	}
	return append(slice, a)
}
