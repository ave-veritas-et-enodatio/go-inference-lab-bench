package arch

import (
	"reflect"
	"testing"
)

// TestVisionMRopePositions4x4Merge2 pins the position-buffer builder to the
// clip.cpp PROJECTOR_TYPE_QWEN3VL layout (clip.cpp ~lines 3573-3597) for a known
// 4x4 patch grid with spatial merge 2. The expected array is derived BY HAND
// from the reference nested loop:
//
//   merge_ratio=2, pw=ph=4, n_pos=num_patches=16, buffer length 64.
//   Outer tiles (y,x): (0,0),(0,2),(2,0),(2,2); inner dy,dx in {0,1}.
//   Channel-major [y, x, y, x], 16 entries per channel.
func TestVisionMRopePositions4x4Merge2(t *testing.T) {
	got := VisionMRopePositions(4, 4, 2)

	// Channel 0 (y) and channel 2 (y) are identical; channel 1 (x) and
	// channel 3 (x) are identical. Hand-derived from the reference loop.
	chY := []int32{0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3}
	chX := []int32{0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3}

	want := make([]int32, 0, 64)
	want = append(want, chY...) // channel 0: y
	want = append(want, chX...) // channel 1: x
	want = append(want, chY...) // channel 2: y
	want = append(want, chX...) // channel 3: x

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("position buffer mismatch:\n got  %v\n want %v", got, want)
	}
}

// TestVisionMRopePositionsInvariants checks structural invariants across a range
// of valid grids without hand-tabulating every value: length is 4*n_pos, the
// channels are paired (0==2, 1==3), and all values lie within the grid bounds.
func TestVisionMRopePositionsInvariants(t *testing.T) {
	cases := []struct {
		name                    string
		nx, ny, merge           int
	}{
		{"4x4 merge2", 4, 4, 2},
		{"6x4 merge2", 6, 4, 2},
		{"48x48 merge2", 48, 48, 2},
		{"40x26 merge2", 40, 26, 2}, // 640x416 image / patch16, non-square
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pos := VisionMRopePositions(tc.nx, tc.ny, tc.merge)
			nPos := tc.nx * tc.ny

			if len(pos) != 4*nPos {
				t.Fatalf("length = %d, want %d (4 * %d)", len(pos), 4*nPos, nPos)
			}

			ch := func(c int) []int32 { return pos[c*nPos : (c+1)*nPos] }
			if !reflect.DeepEqual(ch(0), ch(2)) {
				t.Errorf("channel 0 (y) != channel 2 (y)")
			}
			if !reflect.DeepEqual(ch(1), ch(3)) {
				t.Errorf("channel 1 (x) != channel 3 (x)")
			}

			// y values (ch0) in [0, ny), x values (ch1) in [0, nx).
			for i, v := range ch(0) {
				if v < 0 || int(v) >= tc.ny {
					t.Fatalf("y channel value %d at %d out of [0,%d)", v, i, tc.ny)
				}
			}
			for i, v := range ch(1) {
				if v < 0 || int(v) >= tc.nx {
					t.Fatalf("x channel value %d at %d out of [0,%d)", v, i, tc.nx)
				}
			}
		})
	}
}

// TestVisionMRopePositionsContractViolations confirms the defensive guards
// return nil rather than panicking or emitting a malformed buffer.
func TestVisionMRopePositionsContractViolations(t *testing.T) {
	cases := []struct {
		name          string
		nx, ny, merge int
	}{
		{"zero merge", 4, 4, 0},
		{"negative merge", 4, 4, -2},
		{"x not multiple of merge", 5, 4, 2},
		{"y not multiple of merge", 4, 5, 2},
		{"zero grid", 0, 0, 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := VisionMRopePositions(tc.nx, tc.ny, tc.merge); got != nil {
				t.Fatalf("expected nil for invalid input, got %v", got)
			}
		})
	}
}
