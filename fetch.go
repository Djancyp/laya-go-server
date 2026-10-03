package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Models that are too big to embed (Go's linker caps embedded data at 2 GB) are downloaded from
// Hugging Face on first use into <cache dir>/models/<name>/ and kept there.

// remoteFile is one file of a remote model. SHA256 and Size are checked on download, so a changed or
// truncated file (or a hostile mirror) is rejected before it reaches llama.cpp's GGUF parser.
type remoteFile struct {
	Name   string
	Size   int64
	SHA256 string
}

// remoteModel is a GGUF model hosted on Hugging Face, pinned to a commit.
type remoteModel struct {
	Repo  string
	Rev   string // commit hash, never a moving branch
	Files []remoteFile
}

// remoteModels can be selected with LAYA_MODEL like an embedded one; "policy" is the chat model that
// turns a policy into questions for POST /v1/policy.
var remoteModels = map[string]remoteModel{
	"qwen3guard": {"DevQuasar/Qwen.Qwen3Guard-Gen-0.6B-GGUF", "d170d1d53dad98a28df67d1946319c03dec5fb3c", []remoteFile{
		{"Qwen.Qwen3Guard-Gen-0.6B.Q8_0.gguf", 804753472, "ec2c655af07f633bc2a305d097d67c2b88676d969b03dc66de413a5c5c3cb855"}}},
	"policy": {"unsloth/Qwen3-0.6B-GGUF", "50968a4468ef4233ed78cd7c3de230dd1d61a56b", []remoteFile{
		{"Qwen3-0.6B-Q8_0.gguf", 639447744, "e150ed544dfe6016930c026a93913a5e3184181ebfe6ab2223ae01dd0491784c"}}},
}

// cacheRoot is LAYA_CACHE_DIR, or the user cache dir's laya-server.
func cacheRoot() (string, error) {
	if root := os.Getenv("LAYA_CACHE_DIR"); root != "" {
		return root, nil
	}
	base, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("no cache dir: %w (set LAYA_CACHE_DIR)", err)
	}
	return filepath.Join(base, "laya-server"), nil
}

// remoteDir is where the files of remote model name live.
func remoteDir(name string) (string, error) {
	root, err := cacheRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "models", name), nil
}

// ensureRemote downloads the files of rm that are not in dir yet and returns the first one's path.
// A file already there with the expected size is trusted: only verified files are ever renamed into place.
func ensureRemote(ctx context.Context, log *slog.Logger, rm remoteModel, dir string) (string, error) {
	endpoint := strings.TrimRight(cmpOr(os.Getenv("HF_ENDPOINT"), "https://huggingface.co"), "/")
	if err := checkEndpoint(endpoint); err != nil {
		return "", err
	}
	for _, f := range rm.Files {
		dest := filepath.Join(dir, f.Name)
		if st, err := os.Stat(dest); err == nil && st.Size() == f.Size {
			continue
		}
		if err := download(ctx, log, endpoint+"/"+rm.Repo+"/resolve/"+rm.Rev+"/"+f.Name, dest, f); err != nil {
			return "", fmt.Errorf("download %s/%s: %w", rm.Repo, f.Name, err)
		}
	}
	return filepath.Join(dir, rm.Files[0].Name), nil
}

// checkEndpoint requires https (plain http only to a loopback address, for tests and local mirrors).
func checkEndpoint(endpoint string) error {
	u, err := url.Parse(endpoint)
	if err != nil || u.Host == "" {
		return fmt.Errorf("HF_ENDPOINT %q is not a URL", endpoint)
	}
	if u.Scheme == "https" {
		return nil
	}
	if ip := net.ParseIP(u.Hostname()); u.Scheme == "http" && (u.Hostname() == "localhost" || ip != nil && ip.IsLoopback()) {
		return nil
	}
	return fmt.Errorf("HF_ENDPOINT %q: only https (or http to localhost) is allowed", endpoint)
}

func cmpOr(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// download fetches url to dest through a temp file and renames it into place only when the whole
// body arrived with the expected size and SHA-256. HF_TOKEN, if set, authorizes the request (private
// or gated repos); Go does not forward it to a different host after a redirect.
func download(ctx context.Context, log *slog.Logger, url, dest string, want remoteFile) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	if tok := os.Getenv("HF_TOKEN"); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := (&http.Client{}).Do(req) // no overall timeout: the file is large; ctx cancels it
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s", resp.Status)
	}
	if resp.ContentLength >= 0 && resp.ContentLength != want.Size {
		return fmt.Errorf("server announces %d bytes, expected %d", resp.ContentLength, want.Size)
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	out, err := os.CreateTemp(filepath.Dir(dest), filepath.Base(dest)+".*.tmp")
	if err != nil {
		return err
	}
	tmp := out.Name()
	defer os.Remove(tmp) // no-op after the rename

	log.Info("downloading model", "url", url, "bytes", want.Size)
	start := time.Now()
	h := sha256.New()
	body := io.LimitReader(&progress{r: resp.Body, total: want.Size, log: log, name: filepath.Base(dest)}, want.Size+1) // never write more than expected
	n, err := io.Copy(io.MultiWriter(out, h), body)
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	switch {
	case err != nil:
		return err
	case n != want.Size:
		return fmt.Errorf("got %d bytes, expected %d", n, want.Size)
	case hex.EncodeToString(h.Sum(nil)) != want.SHA256:
		return errors.New("SHA-256 mismatch: the file is not the pinned revision")
	}
	if err := os.Chmod(tmp, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, dest); err != nil {
		return err
	}
	log.Info("model downloaded", "file", filepath.Base(dest), "bytes", n, "seconds", int(time.Since(start).Seconds()))
	return nil
}

// progress logs every 10% of a download.
type progress struct {
	r     io.Reader
	total int64
	read  int64
	next  int64 // percent
	log   *slog.Logger
	name  string
}

func (p *progress) Read(b []byte) (int, error) {
	n, err := p.r.Read(b)
	p.read += int64(n)
	if p.total > 0 {
		if pct := p.read * 100 / p.total; pct >= p.next+10 {
			p.next = pct / 10 * 10
			p.log.Info("downloading model", "file", p.name, "percent", p.next)
		}
	}
	return n, err
}
