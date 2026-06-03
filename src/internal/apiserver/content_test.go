package apiserver

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"hash/crc32"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// makePNG returns a tiny solid-color PNG as a data URI string.
func makePNG(t *testing.T, w, h int, c color.Color) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("png encode: %v", err)
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

func makeJPEG(t *testing.T, w, h int) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 80}); err != nil {
		t.Fatalf("jpeg encode: %v", err)
	}
	return "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

func TestMessageContent_UnmarshalString(t *testing.T) {
	var c messageContent
	if err := json.Unmarshal([]byte(`"hello world"`), &c); err != nil {
		t.Fatalf("unmarshal string: %v", err)
	}
	if c.IsMultiPart() {
		t.Errorf("plain string should not be multi-part")
	}
	if c.TextOnly != "hello world" {
		t.Errorf("TextOnly = %q, want %q", c.TextOnly, "hello world")
	}
}

func TestMessageContent_UnmarshalPartsArray(t *testing.T) {
	uri := makePNG(t, 2, 2, color.RGBA{255, 0, 0, 255})
	raw := []byte(`[
		{"type": "text", "text": "describe this"},
		{"type": "image_url", "image_url": {"url": "` + uri + `"}},
		{"type": "text", "text": " in detail"}
	]`)
	var c messageContent
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatalf("unmarshal parts: %v", err)
	}
	if !c.IsMultiPart() {
		t.Errorf("parts array should be multi-part")
	}
	if c.TextOnly != "describe this in detail" {
		t.Errorf("TextOnly = %q, want %q", c.TextOnly, "describe this in detail")
	}
	if got := c.ImageCount(); got != 1 {
		t.Errorf("ImageCount = %d, want 1", got)
	}
	if got := len(c.Parts); got != 3 {
		t.Errorf("len(Parts) = %d, want 3", got)
	}
}

func TestMessageContent_UnmarshalRejectsObject(t *testing.T) {
	var c messageContent
	err := json.Unmarshal([]byte(`{"text": "oops"}`), &c)
	if err == nil {
		t.Fatalf("expected error decoding object as content")
	}
}

func TestExtractImages_PNG(t *testing.T) {
	uri := makePNG(t, 4, 4, color.RGBA{0, 128, 255, 255})
	c := messageContent{Parts: []contentPart{
		{Type: "image_url", ImageURL: &imageURLValue{URL: uri}},
	}}
	imgs, err := c.extractImages()
	if err != nil {
		t.Fatalf("extractImages: %v", err)
	}
	if len(imgs) != 1 {
		t.Fatalf("got %d images, want 1", len(imgs))
	}
	if imgs[0].MIME != "image/png" {
		t.Errorf("MIME = %q, want image/png", imgs[0].MIME)
	}
	b := imgs[0].Image.Bounds()
	if b.Dx() != 4 || b.Dy() != 4 {
		t.Errorf("image bounds = %v, want 4x4", b)
	}
}

func TestExtractImages_JPEG(t *testing.T) {
	uri := makeJPEG(t, 8, 8)
	c := messageContent{Parts: []contentPart{
		{Type: "image_url", ImageURL: &imageURLValue{URL: uri}},
	}}
	imgs, err := c.extractImages()
	if err != nil {
		t.Fatalf("extractImages: %v", err)
	}
	if len(imgs) != 1 || imgs[0].MIME != "image/jpeg" {
		t.Errorf("unexpected result: %+v", imgs)
	}
}

func TestExtractImages_RejectsHTTPURL(t *testing.T) {
	c := messageContent{Parts: []contentPart{
		{Type: "image_url", ImageURL: &imageURLValue{URL: "https://example.com/cat.png"}},
	}}
	_, err := c.extractImages()
	if err == nil || !strings.Contains(err.Error(), "data:") {
		t.Errorf("expected data: URI error, got %v", err)
	}
}

func TestExtractImages_RejectsUnsupportedMIME(t *testing.T) {
	uri := "data:image/webp;base64," + base64.StdEncoding.EncodeToString([]byte("fakewebp"))
	c := messageContent{Parts: []contentPart{
		{Type: "image_url", ImageURL: &imageURLValue{URL: uri}},
	}}
	_, err := c.extractImages()
	if err == nil || !strings.Contains(err.Error(), "image/webp") {
		t.Errorf("expected unsupported MIME error, got %v", err)
	}
}

func TestExtractImages_RejectsNonBase64Encoding(t *testing.T) {
	uri := "data:image/png,not-base64"
	c := messageContent{Parts: []contentPart{
		{Type: "image_url", ImageURL: &imageURLValue{URL: uri}},
	}}
	_, err := c.extractImages()
	if err == nil || !strings.Contains(err.Error(), "base64") {
		t.Errorf("expected base64 error, got %v", err)
	}
}

func TestExtractImages_RejectsMIMEMismatch(t *testing.T) {
	// Declare PNG, ship JPEG bytes — sanity check is supposed to catch this.
	jpegURI := makeJPEG(t, 4, 4)
	swapped := strings.Replace(jpegURI, "data:image/jpeg;", "data:image/png;", 1)
	c := messageContent{Parts: []contentPart{
		{Type: "image_url", ImageURL: &imageURLValue{URL: swapped}},
	}}
	_, err := c.extractImages()
	if err == nil || !strings.Contains(err.Error(), "decoded as") {
		t.Errorf("expected MIME-mismatch error, got %v", err)
	}
}

func TestExtractImages_RejectsUnknownPartType(t *testing.T) {
	c := messageContent{Parts: []contentPart{
		{Type: "audio_url"},
	}}
	_, err := c.extractImages()
	if err == nil || !strings.Contains(err.Error(), "unsupported part type") {
		t.Errorf("expected unsupported-type error, got %v", err)
	}
}

func TestExtractImages_TextOnlyParts(t *testing.T) {
	c := messageContent{Parts: []contentPart{
		{Type: "text", Text: "no images here"},
	}}
	imgs, err := c.extractImages()
	if err != nil {
		t.Fatalf("text-only parts should not error: %v", err)
	}
	if len(imgs) != 0 {
		t.Errorf("got %d images, want 0", len(imgs))
	}
}

// TestExtractImages_RealImageFromDisk round-trips the committed sample
// image (`test_data/vision_test.png`) through the data-URI path. The
// synthetic tests above stay isolated from the filesystem and from real
// image data; this one ensures the decode pipeline handles a true camera-
// shape image (1200×796 in the asset committed at the time of writing).
// The asset is PNG (deterministic decode across implementations — JPEG
// decode can differ at the low bit between decoders, which would be a
// confound in the bench↔llama vision-equivalence comparison); the JPEG
// decode path stays covered by the synthetic makeJPEG test above. The same
// asset feeds the `test_inference.sh --image` / vision-equiv harness, so a
// regression here flags a problem before the harness even runs.
func TestExtractImages_RealImageFromDisk(t *testing.T) {
	// Walk up from the package dir to the repo root to locate test_data/.
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	var asset string
	for i := 0; i < 6; i++ {
		candidate := filepath.Join(dir, "test_data", "vision_test.png")
		if _, err := os.Stat(candidate); err == nil {
			asset = candidate
			break
		}
		dir = filepath.Dir(dir)
	}
	if asset == "" {
		t.Skip("test_data/vision_test.png not found relative to package dir; skipping real-image test")
	}
	raw, err := os.ReadFile(asset)
	if err != nil {
		t.Fatalf("read %s: %v", asset, err)
	}
	uri := "data:image/png;base64," + base64.StdEncoding.EncodeToString(raw)
	c := messageContent{Parts: []contentPart{
		{Type: "image_url", ImageURL: &imageURLValue{URL: uri}},
	}}
	imgs, err := c.extractImages()
	if err != nil {
		t.Fatalf("extractImages: %v", err)
	}
	if len(imgs) != 1 {
		t.Fatalf("got %d images, want 1", len(imgs))
	}
	if imgs[0].MIME != "image/png" {
		t.Errorf("MIME = %q, want image/png", imgs[0].MIME)
	}
	b := imgs[0].Image.Bounds()
	if b.Dx() <= 0 || b.Dy() <= 0 {
		t.Errorf("decoded bounds = %v, want positive dims", b)
	}
	t.Logf("decoded vision_test.png as %dx%d", b.Dx(), b.Dy())
}

func TestExtractImages_RespectsMaxCount(t *testing.T) {
	uri := makePNG(t, 2, 2, color.RGBA{1, 2, 3, 255})
	parts := make([]contentPart, maxImagesPerRequest+1)
	for i := range parts {
		parts[i] = contentPart{Type: "image_url", ImageURL: &imageURLValue{URL: uri}}
	}
	c := messageContent{Parts: parts}
	_, err := c.extractImages()
	if err == nil || !strings.Contains(err.Error(), "too many images") {
		t.Errorf("expected too-many-images error, got %v", err)
	}
}

// forgedPNG builds a minimal, structurally-valid PNG (signature + IHDR with a
// correct CRC + IEND) whose header DECLARES w×h but carries no pixel data — a
// decompression-bomb stand-in. image.DecodeConfig reads these dimensions
// without allocating a pixel buffer.
func forgedPNG(w, h uint32) []byte {
	var b bytes.Buffer
	b.Write([]byte{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a})
	writeChunk := func(typ string, data []byte) {
		var hdr [4]byte
		binary.BigEndian.PutUint32(hdr[:], uint32(len(data)))
		b.Write(hdr[:])
		b.WriteString(typ)
		b.Write(data)
		var crc [4]byte
		binary.BigEndian.PutUint32(crc[:], crc32.ChecksumIEEE(append([]byte(typ), data...)))
		b.Write(crc[:])
	}
	ihdr := make([]byte, 13)
	binary.BigEndian.PutUint32(ihdr[0:4], w)
	binary.BigEndian.PutUint32(ihdr[4:8], h)
	ihdr[8] = 8 // bit depth
	ihdr[9] = 6 // color type: truecolor + alpha (RGBA)
	// ihdr[10..12] = compression, filter, interlace = 0
	writeChunk("IHDR", ihdr)
	writeChunk("IEND", nil)
	return b.Bytes()
}

func TestExtractImages_RejectsDecompressionBomb(t *testing.T) {
	// A few-hundred-byte file whose header claims 10000×10000 (100 MP), over the
	// ~67 MP cap. The dimension guard must reject it before image.Decode would
	// allocate a multi-hundred-MB pixel buffer.
	uri := "data:image/png;base64," + base64.StdEncoding.EncodeToString(forgedPNG(10000, 10000))
	c := messageContent{Parts: []contentPart{
		{Type: "image_url", ImageURL: &imageURLValue{URL: uri}},
	}}
	_, err := c.extractImages()
	if err == nil || !strings.Contains(err.Error(), "pixel limit") {
		t.Errorf("expected pixel-limit (decompression-bomb) rejection, got %v", err)
	}
}
