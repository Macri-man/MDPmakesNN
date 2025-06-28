package mdplib

import (
	"math"
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
	m.Actions[state] = appendIfMissingAction(m.Actions[state], action)
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
