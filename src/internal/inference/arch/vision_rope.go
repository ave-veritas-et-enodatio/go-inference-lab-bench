package arch

// Vision M-RoPE position-buffer construction for Qwen3-VL style towers.
//
// This mirrors, byte-for-byte, the position buffer that llama.cpp builds and
// feeds to ggml_rope_multi(..., GGML_ROPE_TYPE_VISION, ...). The rotation math
// itself is ggml's (shared by construction via the RopeMulti wrapper); the only
// novel, equiv-sensitive surface on our side is the layout of THIS buffer.
//
// Reference (exact source mirrored here):
//   tools/llama.cpp/tools/mtmd/clip.cpp, set_input switch,
//   case PROJECTOR_TYPE_QWEN3VL (shared with QWEN2VL/GLM4V), ~lines 3573-3597:
//
//     const int merge_ratio = hparams.n_merge;        // 2 for qwen3vl
//     const int pw = image_size_width  / patch_size;  // full patch grid width
//     const int ph = image_size_height / patch_size;  // full patch grid height
//     std::vector<int> positions(n_pos * 4);
//     int ptr = 0;
//     for (int y = 0; y < ph; y += merge_ratio)
//       for (int x = 0; x < pw; x += merge_ratio)
//         for (int dy = 0; dy < 2; dy++)
//           for (int dx = 0; dx < 2; dx++) {
//             positions[                  ptr] = y + dy;  // channel 0: y
//             positions[    num_patches + ptr] = x + dx;  // channel 1: x
//             positions[2 * num_patches + ptr] = y + dy;  // channel 2: y
//             positions[3 * num_patches + ptr] = x + dx;  // channel 3: x
//             ptr++;
//           }
//
// For qwen3vl there is no class embedding, so num_patches == n_pos ==
// nPatchesX * nPatchesY. The buffer is CHANNEL-MAJOR: four contiguous blocks of
// n_pos each, channel order [y, x, y, x]. Within each channel, traversal walks
// the patch grid in 2x2 spatial-merge tile order (the dy,dx inner loop), baking
// the merge grouping into token order so positions line up with the
// merge-grouped patch-embed reshape.
//
// The inner dy,dx loops are hardcoded < 2 in the reference (the 2x2 spatial
// merge), independent of merge_ratio; merge_ratio only sets the outer stride.

// VisionMRopePositions builds the 4-channel M-RoPE position buffer for a
// Qwen3-VL vision tower given the full patch grid dimensions and the spatial
// merge size. nPatchesX/nPatchesY are the patch counts along each axis BEFORE
// merging (pw/ph in clip.cpp). spatialMergeSize is hparams.n_merge (2 for
// qwen3vl). The returned slice has length 4*nPatchesX*nPatchesY, laid out
// channel-major in [y, x, y, x] order, matching clip.cpp exactly.
//
// Both grid dimensions must be positive multiples of spatialMergeSize (the
// reference asserts img dims divide patch_size*2); spatialMergeSize must be
// positive. Violations return nil — the caller (graph build) is expected to
// have validated grid geometry upstream, so this is a defensive contract check.
func VisionMRopePositions(nPatchesX, nPatchesY, spatialMergeSize int) []int32 {
	if spatialMergeSize <= 0 || nPatchesX <= 0 || nPatchesY <= 0 {
		return nil
	}
	if nPatchesX%spatialMergeSize != 0 || nPatchesY%spatialMergeSize != 0 {
		return nil
	}

	nPos := nPatchesX * nPatchesY
	positions := make([]int32, nPos*4)

	ptr := 0
	for y := 0; y < nPatchesY; y += spatialMergeSize {
		for x := 0; x < nPatchesX; x += spatialMergeSize {
			for dy := 0; dy < 2; dy++ {
				for dx := 0; dx < 2; dx++ {
					positions[ptr] = int32(y + dy)          // channel 0: y
					positions[nPos+ptr] = int32(x + dx)     // channel 1: x
					positions[2*nPos+ptr] = int32(y + dy)   // channel 2: y
					positions[3*nPos+ptr] = int32(x + dx)   // channel 3: x
					ptr++
				}
			}
		}
	}
	return positions
}
