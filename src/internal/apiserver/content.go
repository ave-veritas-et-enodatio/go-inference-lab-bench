package apiserver

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg" // registers JPEG decoder for image.Decode
	_ "image/png"  // registers PNG decoder for image.Decode
	"strings"
)

// See ARCHITECTURE.md "Vision / Multimodal Subsystem". This introduces a
// typed-parts content shape on chat completion requests matching the
// OpenAI multimodal chat format. The wire shape per request message is
// either
//
//	{"role": "...", "content": "plain text"}                       // pre-existing
//
// or
//
//	{"role": "...", "content": [
//	    {"type": "text",      "text": "..."},
//	    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
//	]}
//
// The decoder accepts both. Phase 1 *parses* the multi-part form, decodes
// any attached images from data: URIs, and short-circuits with a clear
// error if a request actually carries image content — the encoder /
// projector / splice do not exist yet. Future phases (3, 5, 7) wire the
// decoded images through to the engine.

// Phase 1 limits.
const (
	// maxImagesPerRequest caps how many images a single request may attach,
	// summed across all messages (enforced in the completions handler via a
	// cheap pre-decode ImageCount tally). extractImages applies the same bound
	// to a single message as a local guard. Picked high enough not to
	// constrain reasonable VLM use; low enough that the per-image decode
	// budget below bounds total request work.
	maxImagesPerRequest = 16

	// maxImageDataURIBytes caps the size of an individual decoded data:
	// payload. Tuned for ~20MP PNGs while keeping a malicious request from
	// allocating arbitrarily.
	maxImageDataURIBytes = 32 * 1024 * 1024 // 32 MiB

	// maxImagePixels caps the decoded image's pixel count (width*height) to
	// prevent a decompression bomb: a small compressed file can declare enormous
	// dimensions, and image.Decode allocates a pixel buffer proportional to
	// width*height — which the byte-size limit above does NOT bound. 8192*8192
	// (~67 MP) is far above any real VLM input (the preprocessor downsamples
	// anyway) yet caps the decoded RGBA buffer at ~268 MiB worst case.
	maxImagePixels = 8192 * 8192 // ~67 megapixels
)

// contentPart is a single entry in a multi-part message content array.
// Exactly one of Text / ImageURL is populated, based on Type.
type contentPart struct {
	Type     string         `json:"type"`
	Text     string         `json:"text,omitempty"`
	ImageURL *imageURLValue `json:"image_url,omitempty"`
}

type imageURLValue struct {
	URL string `json:"url"`
}

// messageContent is the typed view of a chat message's content field.
// Exactly one of these is meaningful:
//
//   - Parts == nil: the incoming JSON was a plain string. TextOnly holds it.
//   - Parts != nil: the incoming JSON was a parts array. TextOnly holds the
//     concatenation of just the "text" parts (used for the prompt-size
//     guardrail and for INFO-level logging).
//
// The Parts slice preserves input order; the engine must respect that order
// when splicing image embeddings between text spans.
type messageContent struct {
	TextOnly string
	Parts    []contentPart
}

// UnmarshalJSON accepts either a JSON string or a JSON array of contentPart.
func (c *messageContent) UnmarshalJSON(data []byte) error {
	// Trim leading whitespace to decide between string and array. json.Decode
	// would also work but the dispatch is trivial enough to keep inline.
	trimmed := bytes.TrimLeft(data, " \t\r\n")
	if len(trimmed) == 0 {
		return fmt.Errorf("content: empty value")
	}
	switch trimmed[0] {
	case '"':
		var s string
		if err := json.Unmarshal(data, &s); err != nil {
			return fmt.Errorf("content: invalid string: %w", err)
		}
		c.TextOnly = s
		c.Parts = nil
		return nil
	case '[':
		var parts []contentPart
		if err := json.Unmarshal(data, &parts); err != nil {
			return fmt.Errorf("content: invalid parts array: %w", err)
		}
		c.Parts = parts
		var sb strings.Builder
		for _, p := range parts {
			if p.Type == "text" {
				sb.WriteString(p.Text)
			}
		}
		c.TextOnly = sb.String()
		return nil
	default:
		return fmt.Errorf("content: must be a string or an array of parts")
	}
}

// IsMultiPart reports whether the wire content was a parts array (vs. a
// plain string). A parts array with only text entries is still multi-part.
func (c messageContent) IsMultiPart() bool { return c.Parts != nil }

// ImageCount returns the number of image_url parts in the message.
func (c messageContent) ImageCount() int {
	n := 0
	for _, p := range c.Parts {
		if p.Type == "image_url" {
			n++
		}
	}
	return n
}

// decodedImage is an image extracted from a content part, with its
// position in the parts list preserved so a future splice can interleave
// embeddings with text at the right offset.
type decodedImage struct {
	PartIndex int         // index into messageContent.Parts where the image_url part sits
	MIME      string      // "image/png" or "image/jpeg" (Phase 1 supported set)
	Image     image.Image // decoded pixels
}

// extractImages walks the parts array, decodes each image_url's data: URI,
// and returns the resulting image set. Returns an error on the first
// malformed entry; callers receive a single named failure rather than a
// partial decode.
func (c messageContent) extractImages() ([]decodedImage, error) {
	if len(c.Parts) == 0 {
		return nil, nil
	}
	// Count-before-decode: reject an over-budget message before allocating any
	// pixel buffers. decodeImageDataURI allocates up to maxImagePixels per
	// image, so counting first (ImageCount does not decode) bounds the
	// worst-case allocation to the cap — rather than decoding every part the
	// client sent and only then noticing there are too many.
	if n := c.ImageCount(); n > maxImagesPerRequest {
		return nil, fmt.Errorf("content: too many images (%d, limit %d)", n, maxImagesPerRequest)
	}
	var imgs []decodedImage
	for i, p := range c.Parts {
		switch p.Type {
		case "text":
			continue
		case "image_url":
			if p.ImageURL == nil || p.ImageURL.URL == "" {
				return nil, fmt.Errorf("content.parts[%d]: image_url.url is required", i)
			}
			mime, img, err := decodeImageDataURI(p.ImageURL.URL)
			if err != nil {
				return nil, fmt.Errorf("content.parts[%d]: %w", i, err)
			}
			imgs = append(imgs, decodedImage{PartIndex: i, MIME: mime, Image: img})
		default:
			return nil, fmt.Errorf("content.parts[%d]: unsupported part type %q (allowed: text, image_url)", i, p.Type)
		}
	}
	return imgs, nil
}

// decodeImageDataURI parses an inline `data:` URI into a decoded image.
// Network-fetched URIs (http://, https://, file://) are rejected — Phase 1
// of the vision-input plan deliberately confines image input to the
// request body, with no outbound fetch from the request handler.
func decodeImageDataURI(uri string) (mime string, img image.Image, err error) {
	const prefix = "data:"
	if !strings.HasPrefix(uri, prefix) {
		return "", nil, fmt.Errorf("image_url.url must be a data: URI; network fetch is not supported")
	}
	rest := uri[len(prefix):]
	comma := strings.IndexByte(rest, ',')
	if comma < 0 {
		return "", nil, fmt.Errorf("malformed data URI: missing payload separator")
	}
	header := rest[:comma]
	payload := rest[comma+1:]

	// header is `<mediatype>[;param]*[;base64]`. Phase 1 requires `;base64`
	// — URL-encoded raw payloads are out of scope.
	headerParts := strings.Split(header, ";")
	mime = strings.ToLower(strings.TrimSpace(headerParts[0]))
	base64Encoded := false
	for _, p := range headerParts[1:] {
		if strings.EqualFold(strings.TrimSpace(p), "base64") {
			base64Encoded = true
			break
		}
	}
	if !base64Encoded {
		return "", nil, fmt.Errorf("data URI must use base64 encoding")
	}
	if mime != "image/png" && mime != "image/jpeg" {
		return "", nil, fmt.Errorf("unsupported image MIME type %q (allowed: image/png, image/jpeg)", mime)
	}

	// Pre-flight the decoded size before allocating. base64 expands ~33%
	// over raw bytes, so the encoded form is ~ceil(rawBytes * 4/3). The
	// inverse is rawBytes ≈ encodedBytes * 3/4. The check is a cheap
	// upper-bound gate before doing the actual decode allocation.
	if int64(len(payload))*3/4 > int64(maxImageDataURIBytes) {
		return "", nil, fmt.Errorf("image payload exceeds %d-byte limit", maxImageDataURIBytes)
	}

	raw, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return "", nil, fmt.Errorf("base64 decode failed: %w", err)
	}
	if len(raw) > maxImageDataURIBytes {
		return "", nil, fmt.Errorf("image payload exceeds %d-byte limit", maxImageDataURIBytes)
	}

	// Decompression-bomb guard: read the header dimensions (cheap — no pixel
	// allocation) and reject before image.Decode allocates a buffer sized to
	// width*height. The byte-size checks above bound the compressed payload,
	// not the decoded pixel buffer.
	cfg, _, err := image.DecodeConfig(bytes.NewReader(raw))
	if err != nil {
		return "", nil, fmt.Errorf("image header decode failed: %w", err)
	}
	if int64(cfg.Width)*int64(cfg.Height) > maxImagePixels {
		return "", nil, fmt.Errorf("image dimensions %dx%d exceed %d-pixel limit", cfg.Width, cfg.Height, maxImagePixels)
	}

	img, format, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return "", nil, fmt.Errorf("image decode failed: %w", err)
	}
	// Sanity-check the decoded format matches the declared MIME. Mismatch
	// means a client mislabeled the payload — refuse rather than guess.
	expected := map[string]string{"image/png": "png", "image/jpeg": "jpeg"}[mime]
	if format != expected {
		return "", nil, fmt.Errorf("image decoded as %q but MIME declared %q", format, mime)
	}
	return mime, img, nil
}
