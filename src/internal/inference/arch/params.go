package arch

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"regexp"
	"strconv"
	"strings"
)

// GGUFReader abstracts reading metadata from a GGUF file.
type GGUFReader interface {
	GetU32(key string) (uint32, bool)
	GetF32(key string) (float32, bool)
	GetArrInts(key string) ([]int, bool)
	GetArrBools(key string) ([]bool, bool)
	GetTensorDim(tensorName string, dim int) (int64, bool)
}

// ResolvedParams holds the fully resolved parameter values.
type ResolvedParams struct {
	Ints    map[string]int
	Floats  map[string]float32
	Strings map[string]string
	IntArr  map[string][]int
}

// ResolveParams reads GGUF keys and evaluates derived expressions.
func ResolveParams(def *ArchDef, reader GGUFReader) (*ResolvedParams, error) {
	rp := &ResolvedParams{
		Ints:    make(map[string]int),
		Floats:  make(map[string]float32),
		Strings: make(map[string]string),
		IntArr:  make(map[string][]int),
	}

	// Pass 1: resolve GGUF keys
	for name, ggufKey := range def.Params.Keys {
		if err := resolveParam(name, ggufKey, reader, rp); err != nil {
			return nil, fmt.Errorf("param %q: %w", name, err)
		}
	}

	// Pass 2: evaluate derived expressions (multi-pass for inter-dependencies)
	remaining := make(map[string]string, len(def.Params.Derived))
	for k, v := range def.Params.Derived {
		remaining[k] = v
	}
	for range 10 {
		if len(remaining) == 0 {
			break
		}
		progress := false
		for name, expr := range remaining {
			if err := resolveDerived(name, expr, reader, rp); err == nil {
				delete(remaining, name)
				progress = true
			}
		}
		if !progress {
			names := make([]string, 0, len(remaining))
			for name := range remaining {
				names = append(names, name)
			}
			return nil, fmt.Errorf("cannot resolve derived params (possible cycle): %s",
				strings.Join(names, ", "))
		}
	}

	// Pass 3: apply defaults for params that resolved to 0
	for name, fallback := range def.Params.Defaults {
		if v, ok := rp.Ints[name]; ok && v == 0 {
			if fv, ok := rp.Ints[fallback]; ok {
				rp.Ints[name] = fv
			}
		}
	}

	return rp, nil
}

func resolveParam(name, ggufKey string, reader GGUFReader, rp *ResolvedParams) error {
	// Heuristic: if key doesn't look like a dotted GGUF path, treat as literal
	if !strings.Contains(ggufKey, ".") {
		rp.Strings[name] = ggufKey
		return nil
	}

	// Optional params: GGUF key ending with "?" is silently skipped if not found
	optional := strings.HasSuffix(ggufKey, "?")
	if optional {
		ggufKey = strings.TrimSuffix(ggufKey, "?")
	}

	// Try u32 first (most params are integers)
	if v, ok := reader.GetU32(ggufKey); ok {
		rp.Ints[name] = int(v)
		return nil
	}

	// Try f32
	if v, ok := reader.GetF32(ggufKey); ok {
		rp.Floats[name] = v
		return nil
	}

	// Try int array (e.g. rope_sections, per-layer head counts)
	if v, ok := reader.GetArrInts(ggufKey); ok {
		rp.IntArr[name] = v
		// Also store the first element as a scalar Int for params that can be
		// either scalar or per-layer array in GGUF (e.g. head_count_kv)
		if len(v) > 0 {
			rp.Ints[name] = v[0]
		}
		return nil
	}

	// Try bool array (e.g. sliding_window_pattern) — stored as IntArr (0/1)
	if v, ok := reader.GetArrBools(ggufKey); ok {
		arr := make([]int, len(v))
		for i, b := range v {
			if b {
				arr[i] = 1
			}
		}
		rp.IntArr[name] = arr
		if len(arr) > 0 {
			rp.Ints[name] = arr[0]
		}
		return nil
	}

	if optional {
		return nil
	}
	return fmt.Errorf("GGUF key %q not found", ggufKey)
}

// resolveDerived evaluates a simple arithmetic expression or tensor dimension lookup.
func resolveDerived(name, expr string, reader GGUFReader, rp *ResolvedParams) error {
	// Special form: "tensor_name.ne[dim]"
	if strings.Contains(expr, ".ne[") {
		return resolveTensorDim(name, expr, reader, rp)
	}

	// Arithmetic expression: param names, integer literals, + - * / ( )
	val, err := evalExpr(expr, rp)
	if err != nil {
		return err
	}
	rp.Ints[name] = val
	return nil
}

func resolveTensorDim(name, expr string, reader GGUFReader, rp *ResolvedParams) error {
	// Parse "tensor_name.ne[dim]"
	neIdx := strings.Index(expr, ".ne[")
	if neIdx < 0 {
		return fmt.Errorf("invalid tensor dim expression: %s", expr)
	}
	tensorName := strings.TrimSpace(expr[:neIdx])
	dimStr := expr[neIdx+4:]
	dimStr = strings.TrimSuffix(strings.TrimSpace(dimStr), "]")
	dim, err := strconv.Atoi(dimStr)
	if err != nil {
		return fmt.Errorf("invalid dim in %s: %w", expr, err)
	}
	val, ok := reader.GetTensorDim(tensorName, dim)
	if !ok {
		return fmt.Errorf("tensor %q dim %d not found", tensorName, dim)
	}
	rp.Ints[name] = int(val)
	return nil
}

// evalExpr evaluates a simple integer arithmetic expression.
// Supports: param names, integer literals, +, -, *, /, parentheses.
// Uses Go's parser for tokenization.
func evalExpr(expr string, rp *ResolvedParams) (int, error) {
	node, err := parser.ParseExpr(expr)
	if err != nil {
		return 0, fmt.Errorf("parse expression %q: %w", expr, err)
	}
	return evalAST(node, func(name string) (int, error) {
		if v, ok := rp.Ints[name]; ok {
			return v, nil
		}
		return 0, fmt.Errorf("unknown param %q", name)
	})
}

// evalAST recursively evaluates an integer AST expression. identFn resolves
// bare identifiers; pass a rejecting function for fully-expanded expressions.
// Supports: integer literals, +, -, *, /, %, ==, !=, <, >, <=, >=, parentheses.
func evalAST(node ast.Expr, identFn func(string) (int, error)) (int, error) {
	switch n := node.(type) {
	case *ast.BasicLit:
		if n.Kind != token.INT {
			return 0, fmt.Errorf("unsupported literal type: %s", n.Kind)
		}
		return strconv.Atoi(n.Value)

	case *ast.Ident:
		return identFn(n.Name)

	case *ast.BinaryExpr:
		left, err := evalAST(n.X, identFn)
		if err != nil {
			return 0, err
		}
		right, err := evalAST(n.Y, identFn)
		if err != nil {
			return 0, err
		}
		switch n.Op {
		case token.ADD:
			return left + right, nil
		case token.SUB:
			return left - right, nil
		case token.MUL:
			return left * right, nil
		case token.QUO:
			if right == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			return left / right, nil
		case token.REM:
			if right == 0 {
				return 0, fmt.Errorf("modulo by zero")
			}
			return left % right, nil
		case token.EQL:
			return boolInt(left == right), nil
		case token.NEQ:
			return boolInt(left != right), nil
		case token.LSS:
			return boolInt(left < right), nil
		case token.GTR:
			return boolInt(left > right), nil
		case token.LEQ:
			return boolInt(left <= right), nil
		case token.GEQ:
			return boolInt(left >= right), nil
		default:
			return 0, fmt.Errorf("unsupported operator: %s", n.Op)
		}

	case *ast.ParenExpr:
		return evalAST(n.X, identFn)

	default:
		return 0, fmt.Errorf("unsupported expression type: %T", node)
	}
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// Builtin variables available in routing rule expressions via @{name} syntax.
var routingBuiltins = map[string]bool{
	"layer_idx": true,
}

// EvalRoutingRule evaluates a layer routing expression for a given layer index.
// Param references use ${name} syntax and are expanded to their integer values
// before parsing. Builtins (prefixed with _) provide contextual values.
// Supports: @{layer_idx}, ${param}, integer literals, +, -, *, /, %, ==, !=, parentheses.
// Returns true if the expression evaluates to non-zero (truthy).
func EvalRoutingRule(rule string, layerIdx int, rp *ResolvedParams) (bool, error) {
	// Expand @{builtin} references
	builtinValues := map[string]int{"layer_idx": layerIdx}
	expanded := expandRefs(rule, '@', builtinValues)
	// Expand ${param} references
	expanded = expandRefs(expanded, '$', rp.Ints)

	node, err := parser.ParseExpr(expanded)
	if err != nil {
		return false, fmt.Errorf("parse routing rule %q (expanded: %q): %w", rule, expanded, err)
	}
	// After expansion, no identifiers should remain — all are replaced with integer literals
	rejectIdent := func(name string) (int, error) {
		return 0, fmt.Errorf("unexpanded identifier %q (builtins use @{name}, params use ${name})", name)
	}
	val, err := evalAST(node, rejectIdent)
	if err != nil {
		return false, err
	}
	return val != 0, nil
}

// ValidateRoutingExpr checks that a routing rule expression is syntactically valid
// and only references known builtins and ${param} references (which should already
// be expanded or validated separately). Returns nil if valid.
func ValidateRoutingExpr(rule string, declaredParams map[string]bool) []string {
	// Expand both @{builtin} and ${param} to dummy values so the expression parses as valid Go
	dummy := builtinRefRe.ReplaceAllString(rule, "0")
	dummy = paramRefRe.ReplaceAllString(dummy, "0")

	node, err := parser.ParseExpr(dummy)
	if err != nil {
		return []string{fmt.Sprintf("syntax error: %v", err)}
	}
	var errs []string
	validateExprIdents(node, &errs)
	return errs
}

var builtinRefRe = regexp.MustCompile(`@\{[^}]+\}`)
var paramRefRe = regexp.MustCompile(`\$\{[^}]+\}`)

func validateExprIdents(node ast.Expr, errs *[]string) {
	switch n := node.(type) {
	case *ast.Ident:
		*errs = append(*errs, fmt.Sprintf("unknown identifier %q (builtins use @{name}, params use ${name})", n.Name))
	case *ast.BinaryExpr:
		validateExprIdents(n.X, errs)
		validateExprIdents(n.Y, errs)
	case *ast.ParenExpr:
		validateExprIdents(n.X, errs)
	case *ast.BasicLit:
		// ok: integer literals
	default:
		*errs = append(*errs, fmt.Sprintf("unsupported expression node: %T", node))
	}
}

// expandRefs replaces sigil{name} with the integer value from the values map.
// sigil is '$' for param refs or '@' for builtin refs.
func expandRefs(expr string, sigil byte, values map[string]int) string {
	var b strings.Builder
	for i := 0; i < len(expr); i++ {
		if i+1 < len(expr) && expr[i] == sigil && expr[i+1] == '{' {
			end := strings.IndexByte(expr[i+2:], '}')
			if end >= 0 {
				name := expr[i+2 : i+2+end]
				if v, ok := values[name]; ok {
					b.WriteString(strconv.Itoa(v))
				} else {
					b.WriteString(expr[i : i+2+end+1]) // leave unexpanded
				}
				i += 2 + end // skip past }
				continue
			}
		}
		b.WriteByte(expr[i])
	}
	return b.String()
}


// GetInt returns an integer param or an error.
func (rp *ResolvedParams) GetInt(name string) (int, error) {
	if v, ok := rp.Ints[name]; ok {
		return v, nil
	}
	return 0, fmt.Errorf("param %q not found (or not integer)", name)
}

// GetFloat returns a float param or an error.
func (rp *ResolvedParams) GetFloat(name string) (float32, error) {
	if v, ok := rp.Floats[name]; ok {
		return v, nil
	}
	return 0, fmt.Errorf("param %q not found (or not float)", name)
}

// GetString returns a string param or an error.
func (rp *ResolvedParams) GetString(name string) (string, error) {
	if v, ok := rp.Strings[name]; ok {
		return v, nil
	}
	return "", fmt.Errorf("param %q not found (or not string)", name)
}

// GetIntArr returns an int array param or an error.
func (rp *ResolvedParams) GetIntArr(name string) ([]int, error) {
	if v, ok := rp.IntArr[name]; ok {
		return v, nil
	}
	return nil, fmt.Errorf("param %q not found (or not int array)", name)
}
