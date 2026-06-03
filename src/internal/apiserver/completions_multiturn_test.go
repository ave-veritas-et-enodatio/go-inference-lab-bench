package apiserver

import (
	"image"
	"testing"

	"inference-lab-bench/internal/inference"
)

// stubImage returns a distinguishable image.Image whose bounds width encodes
// an identity tag, so flattenImages ordering can be asserted by reading back
// the tag from each result element. Using bounds (cheap, no decode) keeps the
// test focused on the ordering logic rather than image decoding, which
// content_test.go already covers.
func stubImage(tag int) image.Image {
	return image.NewRGBA(image.Rect(0, 0, tag, 1))
}

func tagOf(img image.Image) int { return img.Bounds().Dx() }

// TestFlattenImages_CrossTurnOrder is the highest-bug-risk path: when several
// messages each carry images, the flattened params.Images order must be
// across-messages-ascending, then within-message by part order — matching the
// order the chat template emits image placeholders. A regression that reversed
// either loop, or dropped a message's images, would change the tag sequence
// and fail here.
func TestFlattenImages_CrossTurnOrder(t *testing.T) {
	// Three messages: msg0 has images tagged [10, 11], msg1 none, msg2 has
	// [12]. extractImages yields them in ascending PartIndex within a message,
	// so the per-message lists already carry within-message order.
	perMessage := [][]decodedImage{
		{{PartIndex: 1, Image: stubImage(10)}, {PartIndex: 3, Image: stubImage(11)}},
		nil,
		{{PartIndex: 0, Image: stubImage(12)}},
	}
	got := flattenImages(perMessage, 3)

	want := []int{10, 11, 12}
	if len(got) != len(want) {
		t.Fatalf("flattenImages returned %d images, want %d", len(got), len(want))
	}
	for i, w := range want {
		if g := tagOf(got[i].Image); g != w {
			t.Errorf("flattened image[%d] tag = %d, want %d (full order = %v)", i, g, w, tagsOf(got))
		}
	}
}

// TestFlattenImages_WithinMessageOrder isolates the within-message ordering:
// a single message with two images must keep them in slice order (the order
// extractImages produced from ascending part indices). A swap of the inner
// loop would invert the tags.
func TestFlattenImages_WithinMessageOrder(t *testing.T) {
	perMessage := [][]decodedImage{
		{{PartIndex: 0, Image: stubImage(20)}, {PartIndex: 2, Image: stubImage(21)}},
	}
	got := flattenImages(perMessage, 2)
	if len(got) != 2 || tagOf(got[0].Image) != 20 || tagOf(got[1].Image) != 21 {
		t.Errorf("within-message order wrong: %v, want [20 21]", tagsOf(got))
	}
}

// TestFlattenImages_NoImages confirms an empty/text-only set flattens to an
// empty slice (the call site only invokes flattenImages when total > 0, but
// the helper must not panic on empty inner lists).
func TestFlattenImages_NoImages(t *testing.T) {
	got := flattenImages([][]decodedImage{nil, nil}, 0)
	if len(got) != 0 {
		t.Errorf("expected no images, got %d", len(got))
	}
}

func tagsOf(imgs []inference.ChatImage) []int {
	out := make([]int, len(imgs))
	for i, img := range imgs {
		out[i] = tagOf(img.Image)
	}
	return out
}
