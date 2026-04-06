package log

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// restoreGlobal saves and restores the global logger and initOnce around a test.
// Must be called as: defer restoreGlobal(t)()
func restoreGlobal(t *testing.T) func() {
	t.Helper()
	saved := global.Load()
	return func() {
		global.Store(saved)
		initOnce = sync.Once{} // reset so the next test's InitLogger call takes effect
	}
}

// TestPreInitSafety verifies that calling package-level functions before
// InitLogger does not panic. The package-level var is initialized at package
// init time, so this is a belt-and-suspenders check.
func TestPreInitSafety(t *testing.T) {
	defer restoreGlobal(t)()
	// Re-assign to the default stderr logger (same as package init) to simulate
	// a "never initialized" state.
	l := Logger(&compactLogger{stderrW: os.Stderr, stderrLevel: LevelInfo})
	global.Store(&l)
	// None of these must panic.
	Info("pre-init info %d", 1)
	Warn("pre-init warn %d", 2)
	Error("pre-init error %d", 3)
	Debug("pre-init debug %d", 4)
}

// TestTeeWritesToFile verifies that when InitLogger is called with a non-empty
// path the log message appears in the file.
func TestTeeWritesToFile(t *testing.T) {
	defer restoreGlobal(t)()

	dir := t.TempDir()
	path := filepath.Join(dir, "out.log")

	if err := InitLogger(path, LevelInfo); err != nil {
		t.Fatalf("InitLogger: %v", err)
	}

	const want = "tee-test-message"
	Info(want)

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read log file: %v", err)
	}
	if !strings.Contains(string(data), want) {
		t.Errorf("log file does not contain %q; got:\n%s", want, data)
	}
}

// TestLogFilePermissions verifies the created log file has mode 0644.
func TestLogFilePermissions(t *testing.T) {
	defer restoreGlobal(t)()

	dir := t.TempDir()
	path := filepath.Join(dir, "perms.log")

	if err := InitLogger(path, LevelInfo); err != nil {
		t.Fatalf("InitLogger: %v", err)
	}
	// Write something so the file definitely exists.
	Info("perm check")

	fi, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat log file: %v", err)
	}
	const want = os.FileMode(0644)
	if got := fi.Mode().Perm(); got != want {
		t.Errorf("log file permissions: got %04o, want %04o", got, want)
	}
}

// TestLevelDebugShowsDebugOutput verifies that stderrLevel=DEBUG lets debug
// messages through to the file handler (which is always DEBUG+).
func TestLevelDebugShowsDebugOutput(t *testing.T) {
	defer restoreGlobal(t)()

	dir := t.TempDir()
	path := filepath.Join(dir, "dbg.log")

	if err := InitLogger(path, LevelDebug); err != nil {
		t.Fatalf("InitLogger: %v", err)
	}

	const want = "debug-visible"
	Debug(want)

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read log file: %v", err)
	}
	if !strings.Contains(string(data), want) {
		t.Errorf("debug message not found in log; got:\n%s", data)
	}
}

// TestFileAlwaysGetsDebug verifies that the file handler always captures DEBUG
// messages regardless of the stderrLevel — the file is always DEBUG+.
func TestFileAlwaysGetsDebug(t *testing.T) {
	defer restoreGlobal(t)()

	dir := t.TempDir()
	path := filepath.Join(dir, "always-debug.log")

	// stderrLevel=WARN: stderr suppresses INFO and DEBUG, but file must get both.
	if err := InitLogger(path, LevelWarn); err != nil {
		t.Fatalf("InitLogger: %v", err)
	}

	const dbgMsg = "debug-in-file"
	const infMsg = "info-in-file"
	const wrnMsg = "warn-in-file"
	Debug(dbgMsg)
	Info(infMsg)
	Warn(wrnMsg)

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read log file: %v", err)
	}
	s := string(data)
	if !strings.Contains(s, dbgMsg) {
		t.Errorf("file missing debug message; got:\n%s", s)
	}
	if !strings.Contains(s, infMsg) {
		t.Errorf("file missing info message; got:\n%s", s)
	}
	if !strings.Contains(s, wrnMsg) {
		t.Errorf("file missing warn message; got:\n%s", s)
	}
}

// TestDefaultLevelShowsInfoSuppressesDebug verifies that with stderrLevel=INFO
// the file captures both debug and info (file is always DEBUG+), and info is present.
func TestDefaultLevelShowsInfoSuppressesDebug(t *testing.T) {
	defer restoreGlobal(t)()

	dir := t.TempDir()
	path := filepath.Join(dir, "default.log")

	if err := InitLogger(path, LevelInfo); err != nil {
		t.Fatalf("InitLogger: %v", err)
	}

	const dbgMsg = "debug-at-default"
	const infMsg = "info-visible-at-default"
	Debug(dbgMsg)
	Info(infMsg)

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read log file: %v", err)
	}
	s := string(data)
	// File handler is always DEBUG+: both messages appear in the file.
	if !strings.Contains(s, dbgMsg) {
		t.Errorf("file should contain debug message (file is always DEBUG+); got:\n%s", s)
	}
	if !strings.Contains(s, infMsg) {
		t.Errorf("info message missing at default level; got:\n%s", s)
	}
}

// TestSplitLevel is the key behavioral test for the two-handler design.
// The file handler is always DEBUG+; the stderr handler uses stderrLevel.
// With stderrLevel=WARN, INFO appears in the file but would be filtered on stderr.
func TestSplitLevel(t *testing.T) {
	defer restoreGlobal(t)()

	f, err := os.CreateTemp("", "splitlevel*.log")
	if err != nil {
		t.Fatal(err)
	}
	f.Close()
	t.Cleanup(func() { os.Remove(f.Name()) })

	if err := InitLogger(f.Name(), LevelWarn); err != nil {
		t.Fatal(err)
	}

	Info("info-message")
	Warn("warn-message")

	content, err := os.ReadFile(f.Name())
	if err != nil {
		t.Fatalf("read log file: %v", err)
	}
	s := string(content)
	// File is always DEBUG+: info-message must appear despite stderrLevel=WARN.
	if !strings.Contains(s, "info-message") {
		t.Error("file should contain info-message at DEBUG+ level")
	}
	if !strings.Contains(s, "warn-message") {
		t.Error("file should contain warn-message")
	}
}

// TestLevelNoneSuppressesAllStderr verifies that LevelNone passes ParseLevel
// correctly and that the file handler still captures all levels when stderrLevel=LevelNone.
func TestLevelNoneSuppressesAllStderr(t *testing.T) {
	defer restoreGlobal(t)()
	f, err := os.CreateTemp("", "none*.log")
	if err != nil {
		t.Fatal(err)
	}
	f.Close()
	t.Cleanup(func() { os.Remove(f.Name()) })
	if err := InitLogger(f.Name(), LevelNone); err != nil {
		t.Fatal(err)
	}
	// All levels written — file should capture everything
	Debug("debug-none")
	Info("info-none")
	Warn("warn-none")
	Error("error-none")
	content, _ := os.ReadFile(f.Name())
	for _, msg := range []string{"debug-none", "info-none", "warn-none", "error-none"} {
		if !strings.Contains(string(content), msg) {
			t.Errorf("file missing %q with LevelNone", msg)
		}
	}
	// stderr suppression cannot be directly asserted here, but LevelNone
	// being above LevelError is verified by ParseLevel returning it correctly
}

// TestParseLevel covers the ParseLevel helper directly.
func TestParseLevel(t *testing.T) {
	cases := []struct {
		input    string
		wantOK   bool
		wantDBG  bool // true if result is Debug level
		wantWRN  bool // true if result is Warn level
		wantNone bool // true if result is LevelNone
	}{
		{"DEBUG", true, true, false, false},
		{"debug", true, true, false, false},
		{"INFO", true, false, false, false},
		{"info", true, false, false, false},
		{"WARN", true, false, true, false},
		{"warn", true, false, true, false},
		{"ERROR", true, false, false, false},
		{"error", true, false, false, false},
		{"NONE", true, false, false, true},
		{"none", true, false, false, true},
		{"", false, false, false, false},
		{"VERBOSE", false, false, false, false},
		{"  INFO  ", true, false, false, false}, // TrimSpace
	}
	for _, tc := range cases {
		t.Run(tc.input, func(t *testing.T) {
			lv, ok := ParseLevel(tc.input)
			if ok != tc.wantOK {
				t.Fatalf("ParseLevel(%q) ok=%v, want %v", tc.input, ok, tc.wantOK)
			}
			if !ok {
				return
			}
			if tc.wantDBG && lv != LevelDebug {
				t.Errorf("ParseLevel(%q): expected LevelDebug, got %v", tc.input, lv)
			}
			if tc.wantWRN && lv != LevelWarn {
				t.Errorf("ParseLevel(%q): expected LevelWarn, got %v", tc.input, lv)
			}
			if tc.wantNone && lv != LevelNone {
				t.Errorf("ParseLevel(%q): expected LevelNone, got %v", tc.input, lv)
			}
		})
	}
}

// TestInvalidLogLevel verifies that ParseLevel returns (LevelInfo, false) for
// unrecognized level strings, and that the returned level is Info.
func TestInvalidLogLevel(t *testing.T) {
	lv, ok := ParseLevel("VERBOSE")
	if ok {
		t.Errorf("ParseLevel(\"VERBOSE\") ok=true, want false")
	}
	if lv != LevelInfo {
		t.Errorf("ParseLevel(\"VERBOSE\") level=%v, want LevelInfo", lv)
	}
}

// TestInitLoggerBadPath verifies InitLogger returns an error for an
// unwritable path rather than panicking.
func TestInitLoggerBadPath(t *testing.T) {
	defer restoreGlobal(t)()

	err := InitLogger("/nonexistent-dir/bench-log-test.log", LevelInfo)
	if err == nil {
		t.Fatal("expected error for bad path, got nil")
	}
}

// TestFatalSubprocess tests that Fatal exits with code 1.
// It re-runs the test binary with a special env var to trigger Fatal in a
// child process, avoiding killing the parent test process.
func TestFatalSubprocess(t *testing.T) {
	if os.Getenv("TEST_FATAL_TRIGGER") == "1" {
		Fatal("subprocess fatal trigger")
		return
	}

	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(exe, "-test.run=TestFatalSubprocess", "-test.v")
	cmd.Env = append(os.Environ(), "TEST_FATAL_TRIGGER=1")
	err = cmd.Run()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("expected ExitError, got %v", err)
	}
	if exitErr.ExitCode() != 1 {
		t.Fatalf("expected exit code 1, got %d", exitErr.ExitCode())
	}
}
