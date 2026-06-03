package arch

import (
	"math"
	"testing"
)

// masked reports whether the (query, key) cell is attention-masked (-Inf).
func masked(data []float32, q, k, nKV int64) bool {
	return math.IsInf(float64(data[q*nKV+k]), -1)
}

// TestSWAMask_BidirectionalSpan verifies that a vision span re-opens the
// future cells within itself while preserving the sliding-window (far-past)
// mask — mirroring llama-mtmd's image ubatch (causal_attn=false) layered on a
// STANDARD SWA window, which keeps all future keys but still masks past keys
// beyond n_swa. Without this, Gemma 4's SWA layers (the ISWA majority) deny
// image patches their bidirectional context.
func TestSWAMask_BidirectionalSpan(t *testing.T) {
	const n = 10
	const window = 4
	// Image span occupies positions [2, 8).
	spans := []MaskSpan{{Start: 2, Length: 6}}
	data := buildSWAMaskData(nil, n, n, 0, window, spans)

	// Query at the span start (pos 2) must now attend to FUTURE in-span keys
	// (3..7) that plain SWA would have causally masked.
	for k := int64(3); k < 8; k++ {
		if masked(data, 2, k, n) {
			t.Errorf("query 2 should attend to future in-span key %d (bidirectional), but it is masked", k)
		}
	}
	// ...and still to past keys within the window (0,1,2).
	for k := int64(0); k <= 2; k++ {
		if masked(data, 2, k, n) {
			t.Errorf("query 2 should attend to past key %d within window, but it is masked", k)
		}
	}
	// Keys outside the span and in the future (8,9) stay masked for query 2.
	for k := int64(8); k < n; k++ {
		if !masked(data, 2, k, n) {
			t.Errorf("query 2 should NOT attend to out-of-span future key %d, but it is open", k)
		}
	}

	// Query at the span end (pos 7): the far-past window mask must survive even
	// inside the span. Key 2 is in-span but 7-2=5 >= window(4) → still masked
	// (llama STANDARD SWA masks past beyond n_swa regardless of the span).
	if !masked(data, 7, 2, n) {
		t.Errorf("query 7 should NOT attend to in-span key 2 (distance 5 >= window 4); span re-open must not breach the past window")
	}
	// But in-window past in-span keys (3..7) stay open.
	for k := int64(3); k <= 7; k++ {
		if masked(data, 7, k, n) {
			t.Errorf("query 7 should attend to in-window in-span key %d, but it is masked", k)
		}
	}

	// A query OUTSIDE the span keeps pure causal+window behavior: pos 9 attends
	// to [5,9], nothing else.
	for k := int64(0); k < n; k++ {
		want := k >= 5 && k <= 9
		if masked(data, 9, k, n) == want {
			t.Errorf("query 9 (outside span) key %d: masked=%v, want open=%v", k, masked(data, 9, k, n), want)
		}
	}
}

// TestSWAMask_NoSpansIsPlainCausalWindow guards that the nil-span path is
// byte-identical to the original causal sliding-window mask.
func TestSWAMask_NoSpansIsPlainCausalWindow(t *testing.T) {
	const n = 8
	const window = 3
	data := buildSWAMaskData(nil, n, n, 0, window, nil)
	for q := int64(0); q < n; q++ {
		for k := int64(0); k < n; k++ {
			want := k > q || k < q-window // masked iff future or beyond window
			if masked(data, q, k, n) != want {
				t.Errorf("(q=%d,k=%d) masked=%v, want %v", q, k, masked(data, q, k, n), want)
			}
		}
	}
}

// TestCausalMask_BidirectionalSpan guards the existing causal-path span
// re-open: full upper-triangular within the span opens up.
func TestCausalMask_BidirectionalSpan(t *testing.T) {
	const n = 8
	spans := []MaskSpan{{Start: 1, Length: 5}} // positions [1,6)
	data := buildCausalMaskData(nil, n, n, 0, false, spans)
	// Within the span, every (q,k) pair is open (full bidirectional).
	for q := int64(1); q < 6; q++ {
		for k := int64(1); k < 6; k++ {
			if masked(data, q, k, n) {
				t.Errorf("in-span (q=%d,k=%d) should be open (bidirectional), but masked", q, k)
			}
		}
	}
	// Outside the span, causal still holds: query 6 must not see key 7.
	if !masked(data, 6, 7, n) {
		t.Errorf("query 6 should not attend to future key 7 outside any span")
	}
}
