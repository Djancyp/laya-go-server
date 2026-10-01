# laya-server

HTTP server for the Laya classifier (mmBERT encoder GGUF via llama.cpp + Go decision head).
Same request shape as TypeSafe System One, answered locally.

## Run

The llama.cpp libraries and the model (F16 GGUF, head, tokenizer) are embedded in the binary
(linux/amd64, CPU; ~690 MB). On start they are extracted once to `~/.cache/laya-server/<id>/`.

    make assets                      # once: copy libs + model into assets/ (not in git)
    make build && LAYA_API_KEYS=key1,key2 ./laya-server
    # or: LAYA_API_KEYS=key1 GOEXPERIMENT=simd go run .

The Makefile, Dockerfiles and `.air.toml` build with `GOEXPERIMENT=simd`, which enables the
AVX2+FMA `dot`/`axpy` in the decision head (`laya/dot_simd.go`, ~40% faster head). A plain
`go build` still works but uses the scalar path, as do CPUs without AVX2+FMA (checked at startup).

## Docker

The image is self-contained (libs and both models are embedded, ~2 GB; `laya-guard` runs by default, see [Embedded models](#embedded-models-laya_model)) and runs as a non-root user.
Run `make assets` first: `assets/` is not in git and is copied into the build.

    make docker-build                # tags djanonurer/laya-server:latest (override with DOCKERHUB_USER, VERSION)
    docker run -d --name laya -p 8080:8080 \
      -e LAYA_API_KEYS=key1 \
      -v laya-cache:/var/cache/laya \
      djanonurer/laya-server:latest

    curl localhost:8080/readyz       # 200 once the model is loaded (~4 s)
    curl -s -XPOST localhost:8080/v1/systemone -H 'Authorization: Bearer key1' \
      -d '{"state":"The invoice total is wrong.","questions":{"angry":{"type":"noul","instructions":"Is the customer angry?"}}}'

- The volume keeps the extracted model between restarts (startup then skips extraction).
- Without `LAYA_API_KEYS` the container exits; set `LAYA_ALLOW_NO_AUTH=true` for local testing only.
- Probes: `GET /healthz` (liveness), `GET /readyz` (readiness). `docker stop` drains in-flight requests (up to 30 s).
- Any env var from [Config](#config-env) can be passed with `-e`.

Publish to Docker Hub (`docker login` first):

    make push                        # CPU image :latest
    make push VERSION=1.0.0          # tag :1.0.0
    make push DOCKERHUB_USER=me      # another account

GPU: `make push-gpu` builds `Dockerfile.gpu` (llama.cpp with CUDA, tag `:cuda` or `:<version>-cuda`) and
defaults `LAYA_GPU_LAYERS=99`; run it with `--gpus all`. The GPU image has not been built or run yet.

## API

Drop-in for TypeSafe's `POST /v1/systemone` (same request and response shapes, incl. `usage`;
`model` is accepted and ignored). Auth: `Authorization: Bearer <key>`.

    {"state": "user: what day is it?",          // string, object or array (objects sent as compact JSON text)
     "model": "jev-latest",
     "questions": {
       "tool":  {"type":"choice","instructions":"Which tool?","criteria":{"none":"...","get_date":"..."}},
       "level": {"type":"score","instructions":"How urgent?","criteria":["not at all","very"]},  // 2-10 levels
       "hurry": {"type":"noul","instructions":"Is the user in a hurry?","criteria":{"true":"...","false":"..."}}}}  // criteria optional

Criteria order is model input: keep it stable. Answers: noul `{noul}`; choice `{choice, confidence, probabilities}`;
score `{score, confidence, legend, probabilities}` (index-keyed). Errors: 401, 413, 422, 529 (overloaded, retry with
backoff), 504 (timeout), 500; body `{"error":{"code","message"}}` (TypeSafe's error body is undocumented, so not matched).
`GET /healthz`, `GET /readyz`.

**Cost:** every question encodes its own copy of the state (up to 1024 tokens): ~0.1 s/question with a short
state, ~3 s/question with a 1024-token state on 16 CPU cores. Send only the state a question needs.

## Config (env)

| Var | Default |
| --- | --- |
| `LAYA_LLAMA_LIB` / `LAYA_DIR` | embedded copies; set to use external libs / model files |
| `LAYA_CACHE_DIR` | user cache dir (where embedded files are extracted) |
| `LAYA_THREADS` / `LAYA_CONTEXTS` | GOMAXPROCS ÷ contexts / 1 (more contexts barely help on CPU: the encoder is memory-bound) |
| `LAYA_API_KEYS` | required unless `LAYA_ALLOW_NO_AUTH=true` |
| `LAYA_GGUF` | `laya-multilingual-F16.gguf` (other quantizations need an external `LAYA_DIR`) |
| `LAYA_GPU_LAYERS` | 0 (CPU) |
| `LAYA_ADDR` | `:8080` |
| `LAYA_MAX_INFLIGHT` | 16 running+queued, then 529 |
| `LAYA_REQUEST_TIMEOUT` | 30s |
| `LAYA_MAX_QUESTIONS` / `_STATE_BYTES` | 64 / 65536 |

## Limits

One llama context: predictions run one at a time (~120 ms/question on 16-core CPU). Scale with replicas
behind a load balancer. TLS and rate limiting belong at the proxy (Traefik/nginx).
Not done: metrics endpoint, per-key rate limits.

## Other checkpoints (ModernBERT, English)

The server is not tied to mmBERT: the hidden size comes from the head file, and `tokenizer.json` may be
Gemma-style (Metaspace BPE) or GPT-2-style (ByteLevel BPE, as in ModernBERT). Mount a model dir with
`tokenizer.json`, `laya-multilingual-head.safetensors` (filename is fixed) and a GGUF:

    docker run -d -p 8080:8080 -e LAYA_API_KEYS=key1 \
      -e LAYA_DIR=/model -e LAYA_GGUF=laya-en-F16.gguf -v /path/to/model:/model:ro djanonurer/laya-server:en

Files must be world-readable (the container runs as uid 10001). Tests run against another checkpoint with
`LAYA_TEST_MODEL_DIR=../models/laya-en LAYA_TEST_REF_DIR=testdata/en LAYA_GGUF=laya-en-F16.gguf go test ./laya`
(references from `laya/testdata/gen_ref_en.py`).

## Embedded models (`LAYA_MODEL`)

The binary embeds the libs plus two models under `assets/`: `laya-guard` (finetuned prompt-injection guard,
calibrated; **the default**) and `laya` (stock multilingual). Only the selected model is extracted. Pick one
with `LAYA_MODEL` (default `laya-guard`); the `.gguf` name is detected when the model dir holds exactly one, so
`LAYA_GGUF` is only needed for a dir with several.

    make assets                       # libs + both models; or just `make assets-guard` / `make assets-stock`
    make docker-build                 # :latest runs laya-guard
    make docker-build-stock           # tag :stock, same image contents but runs the stock model by default
    docker run -d -p 8080:8080 -e LAYA_API_KEYS=key1 djanonurer/laya-server:latest                       # guard
    docker run -d -p 8080:8080 -e LAYA_API_KEYS=key1 -e LAYA_MODEL=laya djanonurer/laya-server:latest    # stock, same image
