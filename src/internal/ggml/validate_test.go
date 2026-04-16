package ggml

import (
	"math"
	"testing"
)

// TestValidateRowData_F32 verifies the Go wrapper correctly delegates to the
// C ggml_validate_row_data for F32 buffers, catching NaN and Inf while
// passing finite data.
func TestValidateRowData_F32(t *testing.T) {
	mkBuf := func(vals []float32) []byte {
		n := len(vals) * 4
		buf := make([]byte, n)
		for i, v := range vals {
			bits := math.Float32bits(v)
			buf[i*4+0] = byte(bits)
			buf[i*4+1] = byte(bits >> 8)
			buf[i*4+2] = byte(bits >> 16)
			buf[i*4+3] = byte(bits >> 24)
		}
		return buf
	}

	cases := []struct {
		name string
		vals []float32
		want bool
	}{
		{"finite", []float32{1, 2, 3, -4, 0, 0.5}, true},
		{"empty", nil, true},
		{"nan_head", []float32{float32(math.NaN()), 1, 2}, false},
		{"nan_tail", []float32{1, 2, float32(math.NaN())}, false},
		{"posinf", []float32{1, float32(math.Inf(1)), 2}, false},
		{"neginf", []float32{1, float32(math.Inf(-1)), 2}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := ValidateRowData(TypeF32, mkBuf(c.vals))
			if got != c.want {
				t.Fatalf("ValidateRowData(F32, %v) = %v, want %v", c.vals, got, c.want)
			}
		})
	}
}

// TestValidateRowData_I32 verifies that integer types are a no-op pass
// (ggml_validate_row_data has nothing to check and returns true).
func TestValidateRowData_I32(t *testing.T) {
	buf := make([]byte, 16) // 4 int32 elements, zero bits
	if !ValidateRowData(TypeI32, buf) {
		t.Fatalf("ValidateRowData(I32, zeros) = false, want true")
	}
}

// TestValidateRowData_Q4K verifies that a well-formed Q4_K block passes and a
// block with a NaN d-scale fails. Q4_K layout per block: uint16 d (F16 scale),
// uint16 dmin (F16 min), 12 bytes of scale/min values, 128 bytes of quants.
// Block size = 144 bytes.
func TestValidateRowData_Q4K(t *testing.T) {
	const blockSize = 144
	good := make([]byte, blockSize)
	// d = 1.0 in F16: 0x3c00; dmin = 0.0 in F16: 0x0000.
	good[0] = 0x00
	good[1] = 0x3c
	good[2] = 0x00
	good[3] = 0x00
	if !ValidateRowData(TypeQ4_K, good) {
		t.Fatalf("ValidateRowData(Q4_K, good) = false, want true")
	}

	// Corrupt: d bits = 0x7c01 is a NaN in F16 (exp=all-ones, mantissa nonzero).
	bad := make([]byte, blockSize)
	bad[0] = 0x01
	bad[1] = 0x7c
	if ValidateRowData(TypeQ4_K, bad) {
		t.Fatalf("ValidateRowData(Q4_K, nan_d) = true, want false")
	}
}
