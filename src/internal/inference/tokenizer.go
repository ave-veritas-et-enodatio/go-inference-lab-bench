package inference

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/dlclark/regexp2"
	ggufparser "github.com/gpustack/gguf-parser-go"
	"github.com/nikolalohinski/gonja/v2"
	"github.com/nikolalohinski/gonja/v2/exec"
)

func init() {
	// Suppress gonja debug logging.
	gonja.SetLoggerOutput(io.Discard)
}

// Tokenizer implements BPE tokenization loaded from a GGUF file's metadata.
// Supports both GPT-2 byte-level BPE (Llama, Qwen) and SentencePiece BPE (Gemma).
type Tokenizer struct {
	tokens          []string          // token ID → string
	tokenIDs        map[string]int32  // string → token ID
	mergeRank       map[[2]string]int // merge rule → priority rank
	bos             int32
	eos             int32
	chatTpl         *exec.Template
	preTokenPattern *regexp2.Regexp
	useByteLevel    bool // true for GPT-2 style, false for SentencePiece style
}

// Qwen3 pre-tokenisation regex (tiktoken / cl100k_base pattern).
// Uses dlclark/regexp2 for lookahead and inline flag support.
const preTokenRegex = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

// specialTokenRe matches special token patterns emitted by chat templates:
//   - <|...|>   — Qwen / Llama style (symmetric)
//   - <|word>   — Gemma 4 style (pipe only at start)
//   - <word|>   — Gemma 4 style (pipe only at end)
//   - <word>    — GLM4 <sop> style
//   - </word>   — closing tags (e.g. </think>)
//   - [WORD]    — GLM4 [gMASK] style
var specialTokenRe = regexp.MustCompile(`<\|[^|>\n]*\|>|<\|[a-zA-Z][a-zA-Z0-9_]*>|<[a-zA-Z][a-zA-Z0-9_]*\|>|</[a-z][a-zA-Z0-9_]*>|<[a-z][a-zA-Z0-9_]*>|\[[a-zA-Z][a-zA-Z0-9_]*\]`)

// NewTokenizerFromGGUF loads vocab, merge rules, BOS/EOS IDs, and the chat
// template Jinja2 string from a parsed GGUF file.
// ggufPath is used to re-read raw token strings, working around a TrimSpace
// bug in the gguf-parser library.
func NewTokenizerFromGGUF(f *ggufparser.GGUFFile, ggufPath string) (*Tokenizer, error) {
	kvs := f.Header.MetadataKV

	// ---- Tokens ----
	// Read raw tokens directly from GGUF to avoid gguf-parser's TrimSpace bug
	// which strips whitespace from token strings (e.g. "\n" → "").
	tokens, err := readGGUFTokensRaw(ggufPath)
	if err != nil {
		return nil, fmt.Errorf("reading raw tokens: %w", err)
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty token list in GGUF")
	}

	// ---- Merges ----
	var merges []string
	mergesKV, ok := kvs.Get("tokenizer.ggml.merges")
	if ok {
		merges = mergesKV.ValueArray().ValuesString()
	}

	// Build lookup maps
	tokenIDs := make(map[string]int32, len(tokens))
	for i, tok := range tokens {
		tokenIDs[tok] = int32(i)
	}

	// Build merge rank map: merge rule string "A B" → rank index
	mergeRank := make(map[[2]string]int, len(merges))
	for rank, m := range merges {
		idx := strings.Index(m, " ")
		if idx < 0 {
			continue
		}
		a, b := m[:idx], m[idx+1:]
		mergeRank[[2]string{a, b}] = rank
	}

	// Read BOS/EOS from GGUF tokenizer metadata
	gt := f.Tokenizer()
	bos := int32(gt.BOSTokenID)
	eos := int32(gt.EOSTokenID)

	// Compile chat template from GGUF.
	// Patch Python slice syntax gonja doesn't support: x[::-1] → x|reverse
	var chatTpl *exec.Template
	if tplKV, ok := kvs.Get("tokenizer.chat_template"); ok {
		src := strings.ReplaceAll(tplKV.ValueString(), "[::-1]", "|reverse")
		tpl, err := gonja.FromString(src)
		if err != nil {
			return nil, fmt.Errorf("compiling chat template: %w", err)
		}
		chatTpl = tpl
	}

	// Compile pre-token regex
	re, err := regexp2.Compile(preTokenRegex, regexp2.Unicode)
	if err != nil {
		return nil, fmt.Errorf("pre-token regex compile: %w", err)
	}

	// Detect tokenizer type: "gpt2" uses byte-level encoding, everything else is SentencePiece.
	useByteLevel := true
	if modelKV, ok := kvs.Get("tokenizer.ggml.model"); ok {
		if modelKV.ValueString() != "gpt2" {
			useByteLevel = false
		}
	}

	return &Tokenizer{
		tokens:          tokens,
		tokenIDs:        tokenIDs,
		mergeRank:       mergeRank,
		bos:             bos,
		eos:             eos,
		chatTpl:         chatTpl,
		preTokenPattern: re,
		useByteLevel:    useByteLevel,
	}, nil
}

// Encode converts a string to token IDs.
// If addSpecial is true, wraps with BOS/EOS.
func (t *Tokenizer) Encode(text string, addSpecial bool) []int32 {
	// 1. Check if the text is itself a special token
	if id, ok := t.tokenIDs[text]; ok {
		return []int32{id}
	}

	// For SentencePiece tokenizers, replace spaces with ▁ (U+2581)
	encText := text
	if !t.useByteLevel {
		encText = strings.ReplaceAll(text, " ", "▁")
	}

	// 2. Pre-tokenise with regex
	var pieces []string
	m, _ := t.preTokenPattern.FindStringMatch(encText)
	for m != nil {
		pieces = append(pieces, m.String())
		m, _ = t.preTokenPattern.FindNextMatch(m)
	}

	// 3. BPE-encode each piece
	var ids []int32
	for _, piece := range pieces {
		ids = append(ids, t.bpePiece(piece)...)
	}

	if addSpecial {
		var wrapped []int32
		if t.bos >= 0 {
			wrapped = append(wrapped, t.bos)
		}
		wrapped = append(wrapped, ids...)
		if t.eos >= 0 {
			wrapped = append(wrapped, t.eos)
		}
		return wrapped
	}
	return ids
}

// Decode converts token IDs to a string.
func (t *Tokenizer) Decode(ids []int32) string {
	var sb strings.Builder
	for _, id := range ids {
		if int(id) < len(t.tokens) {
			tok := t.tokens[id]
			tok = decodeByteLevel(tok)
			sb.WriteString(tok)
		}
	}
	return sb.String()
}

// EncodeChat executes the GGUF chat template via gonja and encodes the result.
// Returns the token ID sequence for the full prompt, ending with the
// assistant turn start marker (ready for completion).
// enableThinking is passed to the template as "enable_thinking" so templates
// that support it can emit or suppress the <think> opening natively.
func (t *Tokenizer) EncodeChat(messages []ChatMessage, enableThinking bool) ([]int32, error) {
	if t.chatTpl == nil {
		return nil, fmt.Errorf("no chat template loaded")
	}

	msgs := make([]map[string]any, len(messages))
	for i, m := range messages {
		msgs[i] = map[string]any{"role": m.Role, "content": m.Content}
	}

	bosStr := ""
	if t.bos >= 0 && int(t.bos) < len(t.tokens) {
		bosStr = t.tokens[t.bos]
	}
	eosStr := ""
	if t.eos >= 0 && int(t.eos) < len(t.tokens) {
		eosStr = t.tokens[t.eos]
	}

	ctx := exec.NewContext(map[string]any{
		"messages":              msgs,
		"add_generation_prompt": true,
		"enable_thinking":       enableThinking,
		"bos_token":             bosStr,
		"eos_token":             eosStr,
	})
	result, err := t.chatTpl.ExecuteToString(ctx)
	if err != nil {
		return nil, fmt.Errorf("chat template render: %w", err)
	}
	return t.encodeWithSpecials(result), nil
}

// ChatMessage is a single message in a conversation.
type ChatMessage struct {
	Role    string
	Content string
}

// VocabContains returns true if s is a direct vocabulary entry (single token).
func (t *Tokenizer) VocabContains(s string) bool {
	_, ok := t.tokenIDs[s]
	return ok
}

// StopID returns the EOS token ID (the generation stop token).
func (t *Tokenizer) StopID() int32 {
	return t.eos
}

// TokenString returns the string for a token ID.
func (t *Tokenizer) TokenString(id int32) string {
	if int(id) >= 0 && int(id) < len(t.tokens) {
		tok := t.tokens[id]
		if t.useByteLevel {
			return decodeByteLevel(tok)
		}
		// SentencePiece: replace ▁ with space
		return strings.ReplaceAll(tok, "▁", " ")
	}
	return ""
}

// encodeWithSpecials encodes a string that may contain special token literals
// (e.g. <|im_start|>, [gMASK], <sop>) mixed with regular text.
// Special token patterns are looked up directly in the vocab; the rest is BPE-encoded.
func (t *Tokenizer) encodeWithSpecials(s string) []int32 {
	var ids []int32
	pos := 0
	for _, loc := range specialTokenRe.FindAllStringIndex(s, -1) {
		if loc[0] > pos {
			ids = append(ids, t.Encode(s[pos:loc[0]], false)...)
		}
		special := s[loc[0]:loc[1]]
		if id, ok := t.tokenIDs[special]; ok {
			ids = append(ids, id)
		} else {
			ids = append(ids, t.Encode(special, false)...)
		}
		pos = loc[1]
	}
	if pos < len(s) {
		ids = append(ids, t.Encode(s[pos:], false)...)
	}
	return ids
}

// ---------------------------------------------------------------------------
// BPE implementation
// ---------------------------------------------------------------------------

func (t *Tokenizer) bpePiece(piece string) []int32 {
	// Convert piece to token-level representation.
	// GPT-2: byte-level encoding (bytes → Unicode codepoints).
	// SentencePiece: raw UTF-8 characters (spaces already replaced with ▁).
	var encoded string
	if t.useByteLevel {
		encoded = encodeByteLevel(piece)
	} else {
		encoded = piece
	}

	// Start with individual characters/bytes as tokens
	syms := strings.Split(encoded, "")
	if len(syms) == 0 {
		return nil
	}

	// Check if the whole piece is a single token
	if id, ok := t.tokenIDs[encoded]; ok {
		return []int32{id}
	}

	// BPE merge loop
	for {
		bestRank := -1
		bestIdx := -1
		for i := 0; i < len(syms)-1; i++ {
			if rank, ok := t.mergeRank[[2]string{syms[i], syms[i+1]}]; ok {
				if bestRank < 0 || rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}
		if bestIdx < 0 {
			break // no more merges possible
		}
		merged := syms[bestIdx] + syms[bestIdx+1]
		newSyms := make([]string, 0, len(syms)-1)
		newSyms = append(newSyms, syms[:bestIdx]...)
		newSyms = append(newSyms, merged)
		newSyms = append(newSyms, syms[bestIdx+2:]...)
		syms = newSyms
	}

	// Convert symbols to IDs
	ids := make([]int32, 0, len(syms))
	for _, sym := range syms {
		if id, ok := t.tokenIDs[sym]; ok {
			ids = append(ids, id)
		}
		// Unknown symbol: could split into byte tokens, skip for now
	}
	return ids
}

// ---------------------------------------------------------------------------
// GPT-2 byte-level BPE encoding/decoding
// ---------------------------------------------------------------------------

// GPT-2 maps raw bytes to specific Unicode codepoints to avoid
// control characters. Bytes 0x00–0xFF are mapped to printable Unicode.
var bytesToUnicode [256]rune
var unicodeToBytes map[rune]byte

func init() {
	// Original GPT-2 byte encoder: 33–126, 161–172, 174–255 are kept as-is.
	// Everything else gets mapped starting at U+0100.
	i := 0
	for b := 0; b < 256; b++ {
		r := rune(b)
		if !((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
			r = rune(0x100 + i)
			i++
		}
		bytesToUnicode[b] = r
	}
	unicodeToBytes = make(map[rune]byte, 256)
	for b, r := range bytesToUnicode {
		unicodeToBytes[r] = byte(b)
	}
}

// encodeByteLevel converts a UTF-8 string to GPT-2 byte-level representation.
func encodeByteLevel(s string) string {
	var sb strings.Builder
	for _, b := range []byte(s) {
		sb.WriteRune(bytesToUnicode[b])
	}
	return sb.String()
}

// decodeByteLevel converts a GPT-2 byte-level token string back to UTF-8.
func decodeByteLevel(s string) string {
	var result []byte
	for _, r := range s {
		if b, ok := unicodeToBytes[r]; ok {
			result = append(result, b)
		} else {
			// Not a GPT-2 byte-level char; encode as UTF-8 (defensive — all
			// 256 byte values are mapped, so this branch is unreachable with
			// valid GPT-2 BPE output).
			var buf [4]byte
			n := utf8.EncodeRune(buf[:], r)
			result = append(result, buf[:n]...)
		}
	}
	return string(result)
}

// readGGUFTokensRaw reads the tokenizer.ggml.tokens string array directly from
// the GGUF file without applying bytes.TrimSpace, working around a bug in the
// gguf-parser library that trims whitespace from all strings.
func readGGUFTokensRaw(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// GGUF header: magic(4) + version(4) + n_tensors(8) + n_kv(8)
	var header struct {
		Magic     uint32
		Version   uint32
		NTensors  uint64
		NKV       uint64
	}
	if err := binary.Read(f, binary.LittleEndian, &header); err != nil {
		return nil, fmt.Errorf("read GGUF header: %w", err)
	}

	readStr := func() (string, error) {
		var length uint64
		if err := binary.Read(f, binary.LittleEndian, &length); err != nil {
			return "", err
		}
		b := make([]byte, length)
		if _, err := io.ReadFull(f, b); err != nil {
			return "", err
		}
		return string(b), nil // no TrimSpace!
	}

	var skipValue func(vtype uint32) error
	skipValue = func(vtype uint32) error {
		switch vtype {
		case 0, 1: // uint8, int8
			_, err := f.Seek(1, io.SeekCurrent)
			return err
		case 2, 3: // uint16, int16
			_, err := f.Seek(2, io.SeekCurrent)
			return err
		case 4, 5, 6: // uint32, int32, float32
			_, err := f.Seek(4, io.SeekCurrent)
			return err
		case 7: // bool
			_, err := f.Seek(1, io.SeekCurrent)
			return err
		case 8: // string
			var length uint64
			if err := binary.Read(f, binary.LittleEndian, &length); err != nil {
				return err
			}
			_, err := f.Seek(int64(length), io.SeekCurrent)
			return err
		case 9: // array
			var atype uint32
			var acount uint64
			if err := binary.Read(f, binary.LittleEndian, &atype); err != nil {
				return err
			}
			if err := binary.Read(f, binary.LittleEndian, &acount); err != nil {
				return err
			}
			for i := uint64(0); i < acount; i++ {
				if err := skipValue(atype); err != nil {
					return err
				}
			}
			return nil
		case 10, 11, 12: // uint64, int64, float64
			_, err := f.Seek(8, io.SeekCurrent)
			return err
		default:
			return fmt.Errorf("unknown GGUF value type %d", vtype)
		}
	}

	// Scan KV pairs looking for tokenizer.ggml.tokens
	for i := uint64(0); i < header.NKV; i++ {
		key, err := readStr()
		if err != nil {
			return nil, fmt.Errorf("read KV key %d: %w", i, err)
		}
		var vtype uint32
		if err := binary.Read(f, binary.LittleEndian, &vtype); err != nil {
			return nil, fmt.Errorf("read KV type %d: %w", i, err)
		}

		if key == "tokenizer.ggml.tokens" && vtype == 9 { // array
			var atype uint32
			var acount uint64
			if err := binary.Read(f, binary.LittleEndian, &atype); err != nil {
				return nil, err
			}
			if atype != 8 { // must be string array
				return nil, fmt.Errorf("tokens array type %d, expected 8 (string)", atype)
			}
			if err := binary.Read(f, binary.LittleEndian, &acount); err != nil {
				return nil, err
			}
			tokens := make([]string, acount)
			for j := uint64(0); j < acount; j++ {
				tokens[j], err = readStr()
				if err != nil {
					return nil, fmt.Errorf("read token %d: %w", j, err)
				}
			}
			return tokens, nil
		}

		if err := skipValue(vtype); err != nil {
			return nil, fmt.Errorf("skip KV value %d (%s): %w", i, key, err)
		}
	}
	return nil, fmt.Errorf("tokenizer.ggml.tokens not found in GGUF")
}
