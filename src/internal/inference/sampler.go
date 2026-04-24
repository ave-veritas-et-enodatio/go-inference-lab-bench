package inference

import (
	"fmt"
	"math"
	"sort"
	"sync/atomic"
)

// ValidateLogits scans a logits slice for NaN or Inf values and returns an
// error identifying the first offending index. Intended as a sanity check at
// the sampling chokepoint: any numerical corruption upstream (bad positional
// encoding, divide-by-zero normalization, overflowing attention) manifests as
// NaN/Inf in the final logits, and catching it here turns a silent "first
// token is EOS / garbage output" symptom into a loud, one-line diagnosis at
// the earliest point the corruption is observable.
//
// Cost: one linear scan of n_vocab (~256KB at 64k vocab) per generated token.
// Memory-bandwidth bound and utterly negligible next to a full forward pass.
func ValidateLogits(logits []float32) error {
	for i, v := range logits {
		f := float64(v)
		if math.IsNaN(f) {
			return fmt.Errorf("logits[%d] is NaN — numerical corruption upstream of sampling", i)
		}
		if math.IsInf(f, 0) {
			return fmt.Errorf("logits[%d] is %+v — numerical overflow upstream of sampling", i, v)
		}
	}
	return nil
}

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

// ComputeTopLogProbs computes log-probabilities for the top-N tokens from raw logits.
// Always includes the chosen token (chosenID). Returns up to topN entries sorted by
// descending probability. tokenString maps token IDs to their string representation.
func ComputeTopLogProbs(logits []float32, chosenID int32, topN int, tokenString func(int32) string) TokenLogProb {
	if topN < 1 {
		topN = 1
	}
	n := len(logits)

	// Stable log-softmax: log(exp(x_i - max) / sum(exp(x_j - max))) = (x_i - max) - log(sum(exp(x_j - max)))
	maxL := logits[0]
	for _, v := range logits {
		if v > maxL {
			maxL = v
		}
	}
	var sumExp float64
	for _, v := range logits {
		sumExp += math.Exp(float64(v - maxL))
	}
	logSumExp := math.Log(sumExp)

	// Find top-N by partial sort: collect (id, logprob) for the N largest logits.
	type entry struct {
		id      int32
		logprob float64
	}
	top := make([]entry, 0, topN+1)
	for i := range n {
		lp := float64(logits[i]-maxL) - logSumExp
		if len(top) < topN {
			top = append(top, entry{int32(i), lp})
			// bubble up
			for j := len(top) - 1; j > 0 && top[j].logprob > top[j-1].logprob; j-- {
				top[j], top[j-1] = top[j-1], top[j]
			}
		} else if lp > top[len(top)-1].logprob {
			top[len(top)-1] = entry{int32(i), lp}
			for j := len(top) - 1; j > 0 && top[j].logprob > top[j-1].logprob; j-- {
				top[j], top[j-1] = top[j-1], top[j]
			}
		}
	}

	// Ensure chosen token is in the list
	chosenLP := float64(logits[chosenID]-maxL) - logSumExp
	found := false
	for _, e := range top {
		if e.id == chosenID {
			found = true
			break
		}
	}
	if !found && len(top) > 0 {
		top[len(top)-1] = entry{chosenID, chosenLP}
		sort.Slice(top, func(i, j int) bool { return top[i].logprob > top[j].logprob })
	}

	// Build result
	chosenStr := tokenString(chosenID)
	result := TokenLogProb{
		ID:      chosenID,
		Token:   chosenStr,
		LogProb: chosenLP,
		Bytes:   ByteArray(chosenStr),
	}
	for _, e := range top {
		s := tokenString(e.id)
		result.TopProbs = append(result.TopProbs, TopLogProb{
			ID:      e.id,
			Token:   s,
			LogProb: e.logprob,
			Bytes:   ByteArray(s),
		})
	}
	return result
}

// Simple LCG PRNG (not crypto-safe; fine for sampling).
//
// lcgState is an atomic so concurrent requests do not produce torn reads or
// writes. Load/Store (not CAS) is intentional: two concurrent samplers
// colliding on the same LCG value produce the same random float, but each
// request has different logits so the collision is statistically invisible in
// output behavior. The goal is memory-model correctness (-race clean), not
// strict per-goroutine sequence uniqueness.
const lcgSeed uint64 = 12345678901234567

var lcgState atomic.Uint64

func init() {
	lcgState.Store(lcgSeed)
}

func pseudoRand() float64 {
	next := lcgState.Load()*6364136223846793005 + 1442695040888963407
	lcgState.Store(next)
	return float64(next>>11) / float64(1<<53)
}
