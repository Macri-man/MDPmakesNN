package nnlib

import (
	"errors"
	"math"
)

// ArgMax returns the index of the maximum value in a slice.
// Returns -1 if the slice is empty.
func ArgMax(vec []float64) int {
	if len(vec) == 0 {
		return -1
	}
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

// Accuracy computes classification accuracy given predicted and target one-hot vectors.
// Returns 0 if inputs are empty or lengths don't match.
func Accuracy(predictions, targets [][]float64) float64 {
	if len(predictions) == 0 || len(targets) == 0 || len(predictions) != len(targets) {
		return 0
	}
	correct := 0
	for i := range predictions {
		if ArgMax(predictions[i]) == ArgMax(targets[i]) {
			correct++
		}
	}
	return float64(correct) / float64(len(predictions))
}

// Sum returns the sum of all elements in a slice.
// Returns 0 for empty slices.
func Sum(vec []float64) float64 {
	sum := 0.0
	for _, v := range vec {
		sum += v
	}
	return sum
}

// Mean computes the arithmetic mean of a slice.
// Returns error if slice is empty.
func Mean(vec []float64) (float64, error) {
	if len(vec) == 0 {
		return 0, errors.New("Mean: input slice is empty")
	}
	return Sum(vec) / float64(len(vec)), nil
}

// Dot computes the dot product of two vectors.
// Returns error if lengths don't match.
func Dot(a, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0, errors.New("Dot: vectors must be the same length")
	}
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum, nil
}

// Add returns element-wise addition of two vectors.
// Returns error if lengths don't match.
func Add(a, b []float64) ([]float64, error) {
	if len(a) != len(b) {
		return nil, errors.New("Add: vectors must be the same length")
	}
	res := make([]float64, len(a))
	for i := range a {
		res[i] = a[i] + b[i]
	}
	return res, nil
}

// Subtract returns element-wise subtraction of two vectors: a - b.
// Returns error if lengths don't match.
func Subtract(a, b []float64) ([]float64, error) {
	if len(a) != len(b) {
		return nil, errors.New("Subtract: vectors must be the same length")
	}
	res := make([]float64, len(a))
	for i := range a {
		res[i] = a[i] - b[i]
	}
	return res, nil
}

// ScalarMultiply multiplies all elements in a vector by a scalar.
func ScalarMultiply(vec []float64, scalar float64) []float64 {
	res := make([]float64, len(vec))
	for i, v := range vec {
		res[i] = v * scalar
	}
	return res
}

// Normalize returns a normalized version of the input vector (unit length).
// Returns error if the input vector is zero-length or norm is zero.
func Normalize(vec []float64) ([]float64, error) {
	norm := 0.0
	for _, v := range vec {
		norm += v * v
	}
	if len(vec) == 0 {
		return nil, errors.New("Normalize: input vector is empty")
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return nil, errors.New("Normalize: zero vector cannot be normalized")
	}
	res := make([]float64, len(vec))
	for i, v := range vec {
		res[i] = v / norm
	}
	return res, nil
}

// Clip limits each element of vec to the range [minVal, maxVal].
func Clip(vec []float64, minVal, maxVal float64) []float64 {
	clipped := make([]float64, len(vec))
	for i, v := range vec {
		if v < minVal {
			clipped[i] = minVal
		} else if v > maxVal {
			clipped[i] = maxVal
		} else {
			clipped[i] = v
		}
	}
	return clipped
}
