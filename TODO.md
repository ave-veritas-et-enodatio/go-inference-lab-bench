# TODO

## Current - Gemma4 Support

### Critical Issues
* None Open

### Serious Issues
* None Open

### Additional Issues
1. Arch SVG diagram
  1. The color legend at the top has an orphaned entry labeled geglu that is SSM attention green
  2. Both attention block types have the same label, color, and internal appearance
    1. These should have appropriate representations of their attention internals 
  3. The geglu SVG snippet doesn't show the proper placement of RMSNorm blocks
    1. Is 'geglu' even the right name? shouldn't that be geglu_attention? 
  4. Routing rule for the two attention block types is not properly shown
2. Layers SVG diagram
  1. The attention heads are all the same type - should be interleaved at appropriate intervals. If the interval comes from a GGUF use a reasonable example value, labeled as such, in the same way the Qwen35 diagrams do.

## Future

### Deferred Work
1. **Structured logging** — replace `fmt.Fprintf(os.Stderr, ...)` with a real logging package (slog or zerolog). Leveled output, consistent format, eliminate Makefile/test_inference.sh stderr redirects.
  1. creating a simple internal logging interface and backing it with a known package would be a good intermediate step
  2. we could back it with a simple implementation and then opt for a full logging package if the delta merited the dependency weight   
2. **Chat client streaming + acontextual mode** — add `--no-history` flag for stateless per-prompt testing with real-time SSE streaming output. Replaces need for test_inference.sh for interactive debugging of thinking models. refactor chat client to separate implementation from CLI entry point
3. Batch inference
4. Multiple concurrent models
5. Linux/CUDA support (rename GPU init, add CUDA backend)
6. **Palette unification** — export a .css color set from the SVG diagram palette (`diagramPalette()`) that the arch-editor can use for block coloring, eliminating the duplicated color constants in editor.js
7. **Diffusion generation loop** — iterative masked denoising for non-causal models (LLaDA-MoE). Architecture definition at `models/arch/llada-moe.arch.toml.nyi`; builder support (attention QK-norm, non-causal mask, MoE FFN) already in place. Needs new generation strategy in engine.go.
8. gated_delta_net.svg should probably be ssm_attention.svg (with that name propagated throughout) shouldn't it?
