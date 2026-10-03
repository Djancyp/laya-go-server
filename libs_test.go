package main

import (
	"bytes"
	"debug/elf"
	"io/fs"
	"strconv"
	"strings"
	"testing"
)

// maxGlibc is the glibc of the runtime image (Dockerfile: debian:bookworm).
const maxGlibc = "2.36"

// newerVersion reports whether dotted version a is above b ("2.43" > "2.36").
func newerVersion(a, b string) bool {
	pa, pb := strings.Split(a, "."), strings.Split(b, ".")
	for i := range max(len(pa), len(pb)) {
		var x, y int
		if i < len(pa) {
			x, _ = strconv.Atoi(pa[i])
		}
		if i < len(pb) {
			y, _ = strconv.Atoi(pb[i])
		}
		if x != y {
			return x > y
		}
	}
	return false
}

// Every embedded lib must find its sibling libs through $ORIGIN (or the default search path). A RUNPATH
// pointing at a build directory works on the build machine only: elsewhere dlopen fails with
// "libggml-base.so.0: cannot open shared object file".
func TestEmbeddedLibsRunpath(t *testing.T) {
	n := 0
	err := fs.WalkDir(assets, "assets", func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasPrefix(p, "assets/llama") || !strings.Contains(d.Name(), ".so") {
			return err
		}
		b, err := fs.ReadFile(assets, p)
		if err != nil {
			return err
		}
		f, err := elf.NewFile(bytes.NewReader(b))
		if err != nil {
			t.Errorf("%s: not an ELF file: %v", p, err)
			return nil
		}
		defer f.Close()
		n++
		// The runtime image is Debian bookworm: a lib built against a newer glibc fails to load there
		// ("version `GLIBC_2.43' not found").
		syms, _ := f.ImportedSymbols()
		for _, sym := range syms {
			if v, ok := strings.CutPrefix(sym.Version, "GLIBC_"); ok && newerVersion(v, maxGlibc) {
				t.Errorf("%s: needs GLIBC_%s (%s), the image has %s; build it in the Dockerfile's bookworm stage (make llama-deberta)", p, v, sym.Name, maxGlibc)
				break
			}
		}
		for _, tag := range []elf.DynTag{elf.DT_RUNPATH, elf.DT_RPATH} {
			paths, _ := f.DynString(tag)
			for _, list := range paths {
				for _, dir := range strings.Split(list, ":") {
					if dir != "" && dir != "$ORIGIN" {
						t.Errorf("%s: %v %q is machine-specific; relink with $ORIGIN", p, tag, dir)
					}
				}
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if n == 0 {
		t.Skip("no embedded libs (run make assets)")
	}
}

func TestNewerVersion(t *testing.T) {
	for _, c := range []struct {
		a, b string
		want bool
	}{{"2.43", "2.36", true}, {"2.36", "2.36", false}, {"2.2.5", "2.36", false}, {"2.4", "2.36", false}, {"3", "2.36", true}} {
		if got := newerVersion(c.a, c.b); got != c.want {
			t.Errorf("newerVersion(%q, %q) = %v", c.a, c.b, got)
		}
	}
}
