# ggml lib

This library is a thin pure C wrapper around the ggml pure C interface.
It limits the surface area ggml exposes to CGO. This is done for two reasons

* simplifies what CGO has to process - saves build time and risk
* simplifies what the go side of the binding needs to wrap to presenting a pure-go API

## Invariants and Aesthetics
* HELPER FUNCTIONS ARE NOT IMPLEMENTED IN C
  * Only types and functions are exposed
  * Types may be obscured by void* typedefs with paramers packed into argument structs in the .c file, but that's it
  * We do not want to be debugging C code in a go project. 
* Isolation preserved
  * The following pieces should always be extractable to make a single, clean unit
    * src/third_party/ggml
    * src/ggml_lib 
    * src/internal/log
    * src/internal/ggm
  * TODO: consider making the log dependency abstract so that only the C++ lib, C wrapper, go package can be separated at will
  * We don't necessarily intend to separate these pieces out as a separate dependency but architectural hygiene dictates it should be kept trivially possible
