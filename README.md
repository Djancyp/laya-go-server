# laya-server

HTTP server for typed classification questions (TypeSafe System One shape), answered locally: encoder GGUFs
through llama.cpp plus a Go decision head. Default model: GLiNER2.5-Decide (`gliner2-decide`); `laya-guard`, the stock
`laya` and Qwen3Guard are selectable with `LAYA_MODEL`. `POST /v1/policy` turns a written policy into questions.

## Run

The llama.cpp libraries and three models (GLiNER2.5-Decide with its patched libs, laya-guard, stock laya) are embedded in the
binary (linux/amd64, CPU; ~1.9 GB, near Go's 2 GB linker cap). On start the libs and the selected model are extracted once to
`~/.cache/laya-server/<id>/`. Models too big to embed are downloaded on first use, see [Downloaded models](#downloaded-models).

    make assets                      # once: copy libs + models into assets/ (not in git)
    make build && LAYA_API_KEYS=key1,key2 ./laya-server
    # or: LAYA_API_KEYS=key1 GOEXPERIMENT=simd go run .

The Makefile, Dockerfiles and `.air.toml` build with `GOEXPERIMENT=simd`, which enables the
AVX2+FMA `dot`/`axpy` in the decision head (`laya/dot_simd.go`, ~40% faster head). A plain
`go build` still works but uses the scalar path, as do CPUs without AVX2+FMA (checked at startup).

## Docker

The image is self-contained (libs and three models are embedded, ~1.9 GB; `gliner2-decide` runs by default, see [Embedded models](#embedded-models-laya_model)) and runs as a non-root user.
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

**Moderation model** (`LAYA_MODEL=qwen3guard`, Qwen3Guard-Gen-0.6B Q8_0, downloaded on first start): `questions` become optional and the reply gains `"moderation":
{"safety":"Safe|Unsafe|Controversial","probabilities":{...},"categories":["Violent",...],"refusal":true}`. `state` is the user
prompt; add `"response": "<assistant reply>"` to assess the reply instead (adds `refusal`). Questions still work: `noul` =
P(unsafe or controversial), `choice` options must be `safe`/`unsafe`/`controversial`. Its policy is fixed (9 categories);
editing the policy text in the prompt did not steer it.

**Policy to questions** (`POST /v1/policy`): body `{"policy": "<policy text>", "target": "request"|"response"}` (`target`
optional, default `request`). Qwen3-0.6B turns every prohibition in the policy into a yes/no question; the reply is
`{"questions": {id: {"type":"noul","instructions":...}}}`: add a `state`, edit if you like, and send it to `/v1/systemone`.
No model runs on any state here. Results are cached per policy (first call a few seconds, then instant). Limits: policy 8 KB,
48 rules, `LAYA_MAX_QUESTIONS` questions. Any plain-English policy converts, however it is written (one line, bullets, a run-on paragraph, no punctuation):
it is split into rules (lines, bullets, sentences, semicolon clauses, requirement lists, "… and never …"), and every rule
becomes a question. A simple negated action ("dont allow anyone to send an email", "never mention competitors") becomes
"Does the text do, or ask someone to do, the following: send an email?"; other prohibitions are rewritten by the model;
anything else ("keep it professional", "requires written approval") becomes "Does the text go against this rule: …?".
Definitions, scope, consequences and pure permissions ("the agent may discuss X") are skipped; a policy with no rule left
becomes one question. English wording works best.
A first call converts every rule with the model (about 0.5 s each, up to ~25 s for 48), hence the longer `LAYA_POLICY_TIMEOUT`. The chat model is downloaded in the background on first start; until it is loaded the
endpoint answers 503 (`Retry-After: 30`), and 501 when it is off (`LAYA_POLICY_MODEL=off`) or failed to load.

**Cost:** every question encodes its own copy of the state (up to 1024 tokens): ~0.1 s/question with a short
state, ~3 s/question with a 1024-token state on 16 CPU cores. Send only the state a question needs.

## Config (env)

| Var | Default |
| --- | --- |
| `LAYA_LLAMA_LIB` / `LAYA_DIR` | embedded copies; set to use external libs / model files |
| `LAYA_CACHE_DIR` | user cache dir (where embedded files are extracted) |
| `LAYA_THREADS` / `LAYA_CONTEXTS` | GOMAXPROCS ÷ contexts / 1 (more contexts barely help on CPU: the encoder is memory-bound) |
| `LAYA_API_KEYS` | required unless `LAYA_ALLOW_NO_AUTH=true` |
| `LAYA_MODEL` | `gliner2-decide` (also `laya-guard`, `laya`, `qwen3guard`) |
| `LAYA_POLICY_MODEL` / `LAYA_POLICY_GGUF` | on / downloaded; `off` disables `/v1/policy`, `LAYA_POLICY_GGUF=/path/Qwen3-0.6B-Q8_0.gguf` uses a local file (offline) |
| `LAYA_POLICY_INFLIGHT` / `LAYA_POLICY_TIMEOUT` | 2 / 120s: `/v1/policy` has its own slots and timeout, so slow policy conversions cannot starve `/v1/systemone` |
| `HF_TOKEN` / `HF_ENDPOINT` | none / `https://huggingface.co` (download auth and mirror; https only, or http to localhost) |
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

The binary embeds the libs plus three models under `assets/`: `gliner2-decide` (GLiNER2.5-Decide, DeBERTa-v3-large Q8_0;
**the default**), `laya-guard` (finetuned prompt-injection guard, calibrated) and `laya` (stock multilingual). Only the
selected model and the libs it runs on are extracted. The `.gguf` name is detected when the model dir holds exactly one, so
`LAYA_GGUF` is only needed for a dir with several.

GLiNER is a DeBERTa-v2 encoder that stock llama.cpp cannot load: it runs on the patched build embedded as
`assets/llama-deberta` (`make assets-gliner` copies it from `GLINER_LLAMA`); the other models use the stock libs in
`assets/llama`. The patched build is portable (`GGML_BACKEND_DL` + all CPU variants, picked at run time like the stock libs), so the
image is not tied to the build machine's CPU. The CUDA image has no patched libs and defaults to `laya-guard`.

    make assets                       # libs + all models; or `make assets-gliner` / `assets-guard` / `assets-stock`
    make docker-build                 # :latest runs gliner2-decide
    make docker-build-stock           # tag :stock, same image contents but runs the stock model by default
    docker run -d -p 8080:8080 -e LAYA_API_KEYS=key1 djanonurer/laya-server:latest                          # gliner2-decide
    docker run -d -p 8080:8080 -e LAYA_API_KEYS=key1 -e LAYA_MODEL=laya-guard djanonurer/laya-server:latest # guard, same image

## Downloaded models

Models over the embed limit are fetched from Hugging Face on first use into `<cache dir>/models/<name>/` (default
`~/.cache/laya-server`; mount `LAYA_CACHE_DIR` on a volume in Docker) and kept: `qwen3guard` (DevQuasar Qwen3Guard-Gen-0.6B,
Q8_0, 0.8 GB, when selected with `LAYA_MODEL`) and the policy chat model (unsloth Qwen3-0.6B Q8_0, 0.6 GB, for `/v1/policy`).
A model download blocks start-up (`LAYA_MODEL=qwen3guard`); the policy model downloads in the background. Downloads are
pinned to a commit and checked against a SHA-256 and size (`remoteModels` in `fetch.go`), and a file appears only once it
passes, so a changed, truncated or tampered file is rejected. `HF_TOKEN` authorizes private repos; `HF_ENDPOINT` points at a mirror.
