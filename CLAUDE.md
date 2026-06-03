# go-inference-lab-bench — working principles
Full contract: AGENTS.md. This file is the drift-watch: rules quietly broken when AGENTS.md falls out of context.

## Read the docs before ad-hoc grepping
Before discovering a capability by grepping source — a harness/API/tooling
feature, where a subsystem lives, how something is wired — check **AGENTS.md**
(API extensions, `test_inference.sh`/`.py` env knobs, `test_*_equiv.sh`, tooling)
and **ARCHITECTURE.md** (subsystem/source map) first. They summarize these; grep
source only to confirm wiring once the docs point you at the right place. Example
miss: hunting apiserver/engine code for how `"stateless"` is passed when it's
listed in AGENTS.md's API extensions — and the documented "stateless is the only
`ForwardCaptures` mode" fact answers *why* the capture path is stateless.

## Build & validation
- Never `go build` directly. Use `make` — it wires ggml, cgo flags, symlinks.
- Acceptance gate, every change: `make test && make integration-test && make equiv-test`. No partial gates.
- Only one `make integration-test` / `make equiv-test` at a time (port + GPU contention). Sequential validation.
- `ALL_MODELS=true bash test_inference.sh ...` before declaring an inference change done. Edge cases surface off Llama.
- Load-time defensive checks (`ResolveParams`, `ValidateRowData`, `ValidateLogits`) stay on. Don't disable without cause.

## Coding
- Use coder agents for coding work unless directed otherwise. Ensure coder agents receive AGENTS.md to understand full contract when working.
- In the cases when you are asked to do direct coding work, read the appropriate coding agent definition and AGENTS.md if it is not fresh in context. It is crucial to maintain the invariants and contracts specified in those documents.

## Conversational tone
Concise. Competent. No unearned praise (e.g. "that's a sharp question" for every query.)
Reserve that language for moments of significant insight, intelligence, capability.

## Task management
- Whenever possible dispatch tasks to sub-agents to remain free for discussion, planning, and other interactive functions. Long spells of unavailability shut out the user's ability to multi-task across the current project needs.
- *INVARIANT* Never continue with a plan or process in response to a user question unless the prompt explicitly says to proceed.
  - If the user has an open question, this must be addressed before proceeding with a plan.
  - Tool use in order to gather data to answer the question is not disallowed unless already restricted by other instructions.
