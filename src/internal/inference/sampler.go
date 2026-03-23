package inference

import (
	"math"
	"sort"
)

// Greedy returns the token ID with the highest logit.
func Greedy(logits []float32) int32 {
	best := int32(0)
	bestVal := logits[0]
	for i, v := range logits {
		if v > bestVal {
			bestVal = v
			best = int32(i)
		}
	}
	return best
}

// TopP samples from the smallest set of tokens whose cumulative probability
// is >= p. Temperature is applied before sampling.
func TopP(logits []float32, temperature, topP float32) int32 {
	if temperature <= 0 || topP <= 0 {
		return Greedy(logits)
	}

	n := len(logits)
	probs := make([]float32, n)

	// Apply temperature and softmax
	maxL := logits[0]
	for _, v := range logits {
		if v > maxL {
			maxL = v
		}
	}
	sum := float32(0)
	for i, v := range logits {
		probs[i] = float32(math.Exp(float64((v - maxL) / temperature)))
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}

	// Sort indices by descending probability
	idxs := make([]int, n)
	for i := range idxs {
		idxs[i] = i
	}
	sort.Slice(idxs, func(a, b int) bool {
		return probs[idxs[a]] > probs[idxs[b]]
	})

	// Find cutoff
	cumul := float32(0)
	cutoff := n
	for i, idx := range idxs {
		cumul += probs[idx]
		if cumul >= topP {
			cutoff = i + 1
			break
		}
	}
	idxs = idxs[:cutoff]

	// Sample
	r := float32(pseudoRand()) * cumul
	cumul2 := float32(0)
	for _, idx := range idxs {
		cumul2 += probs[idx]
		if r < cumul2 {
			return int32(idx)
		}
	}
	return int32(idxs[len(idxs)-1])
}

// Simple LCG PRNG (not crypto-safe; fine for sampling).
var lcgState uint64 = 12345678901234567

func pseudoRand() float64 {
	lcgState = lcgState*6364136223846793005 + 1442695040888963407
	return float64(lcgState>>11) / float64(1<<53)
}
