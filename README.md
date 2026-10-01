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
