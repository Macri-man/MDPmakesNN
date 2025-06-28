package mdplib

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"io"
	"os"
	"strconv"
)

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

		if m.Transitions[s] == nil {
			m.Transitions[s] = make(map[Action][]Transition)
		}
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

		if m.Transitions[s] == nil {
			m.Transitions[s] = make(map[Action][]Transition)
		}
		m.Actions[s] = appendIfMissingAction(m.Actions[s], a)
		m.Transitions[s][a] = append(m.Transitions[s][a], Transition{
			NextState: ns, Prob: entry.Prob, Reward: entry.Reward,
		})
	}
	return nil
}

func appendIfMissing(states []State, s State) []State {
	for _, existing := range states {
		if existing == s {
			return states
		}
	}
	return append(states, s)
}

func appendIfMissingAction(actions []Action, a Action) []Action {
	for _, existing := range actions {
		if existing == a {
			return actions
		}
	}
	return append(actions, a)
}
