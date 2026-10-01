package main

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
)

// The llama.cpp libraries and the Laya model files ship inside the binary (linux/amd64).
// dlopen and llama.cpp need real files, so they are extracted to a cache dir on first start.
// Populate assets/ with `make assets` (CPU libs) or the Dockerfile.gpu build (CUDA libs, -tags cuda).
// The embed itself lives in assets_cpu.go / assets_cuda.go.

// soFile matches a versioned library such as libllama.so.0.4.1; go:embed cannot hold symlinks,
// so the SONAME (libllama.so.0) and dev (libllama.so) links the loader asks for are recreated.
var soFile = regexp.MustCompile(`^(lib.+\.so)\.(\d+)\.\d+\.\d+$`)

// extractAssets writes the embedded files under LAYA_CACHE_DIR (default: the user cache dir),
// in a directory named after the embedded content, and returns the lib and model dirs.
// Files already there with the right size are kept, so restarts are instant.
func extractAssets(src fs.FS) (libDir, modelDir string, err error) {
	root := os.Getenv("LAYA_CACHE_DIR")
	if root == "" {
		base, err := os.UserCacheDir()
		if err != nil {
			return "", "", fmt.Errorf("no cache dir: %w (set LAYA_CACHE_DIR)", err)
		}
		root = filepath.Join(base, "laya-server")
	}
	id, err := assetsID(src)
	if err != nil {
		return "", "", err
	}
	root = filepath.Join(root, id)

	err = fs.WalkDir(src, "assets", func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		return extractFile(src, p, filepath.Join(root, filepath.FromSlash(strings.TrimPrefix(p, "assets/"))))
	})
	if err != nil {
		return "", "", err
	}

	libDir = filepath.Join(root, libAssetsDir)
	entries, err := os.ReadDir(libDir)
	if err != nil {
		return "", "", err
	}
	for _, e := range entries {
		m := soFile.FindStringSubmatch(e.Name())
		if m == nil {
			continue
		}
		for _, link := range []string{m[1] + "." + m[2], m[1]} {
			p := filepath.Join(libDir, link)
			if _, err := os.Lstat(p); err == nil {
				continue
			}
			if err := os.Symlink(e.Name(), p); err != nil && !errors.Is(err, fs.ErrExist) { // another start may win the race
				return "", "", err
			}
		}
	}
	return libDir, filepath.Join(root, "laya"), nil
}

// assetsID names the cache dir after the embedded file names, sizes and the first and last
// sampleBytes of each file: cheap (files are never read in full) yet it changes when a
// model is swapped for one of equal size.
func assetsID(src fs.FS) (string, error) {
	h := sha256.New()
	err := fs.WalkDir(src, "assets", func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		fi, err := d.Info()
		if err != nil {
			return err
		}
		fmt.Fprintf(h, "%s:%d\n", path.Clean(p), fi.Size())
		return sampleFile(h, src, p, fi.Size())
	})
	return fmt.Sprintf("%x", h.Sum(nil)[:6]), err
}

func extractFile(src fs.FS, from, to string) error {
	in, err := src.Open(from)
	if err != nil {
		return err
	}
	defer in.Close()
	fi, err := in.Stat()
	if err != nil {
		return err
	}
	if st, err := os.Stat(to); err == nil && st.Size() == fi.Size() {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(to), 0o755); err != nil {
		return err
	}
	// Write then rename: a crash mid-copy never leaves a file with the right size.
	// A unique temp name keeps concurrent starts sharing one cache dir from clobbering each other.
	out, err := os.CreateTemp(filepath.Dir(to), filepath.Base(to)+".*.tmp")
	if err != nil {
		return err
	}
	tmp := out.Name()
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		os.Remove(tmp)
		return err
	}
	if err := out.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Chmod(tmp, 0o644); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, to); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}

const sampleBytes = 64 << 10

// sampleFile feeds the first and last sampleBytes of p into h.
func sampleFile(h io.Writer, src fs.FS, p string, size int64) error {
	f, err := src.Open(p)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := io.CopyN(h, f, min(size, sampleBytes)); err != nil {
		return err
	}
	sk, ok := f.(io.Seeker)
	if size <= sampleBytes || !ok { // embedded files are seekable; other FSes get the head only
		return nil
	}
	if _, err := sk.Seek(max(sampleBytes, size-sampleBytes), io.SeekStart); err != nil {
		return err
	}
	_, err = io.Copy(h, f)
	return err
}
