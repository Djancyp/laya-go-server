# Source of the files embedded into the binary (see `make assets`). The policy model (POST /v1/policy) and
# Qwen3Guard are not embedded: the server downloads them from Hugging Face on first use.
LLAMA_SRC ?= $(HOME)/.kronk/libraries/linux/amd64/cpu
MODEL_SRC ?= $(HOME)/Documents/codding/experiment/models/laya
# Calibrated prompt-injection guard model (output of scripts/v2/to_gguf.py in the laya-finetune repo).
GUARD_SRC ?= $(HOME)/Documents/codding/experiment/models/laya-guard
# GLiNER2.5-Decide (DeBERTa-v3-large): needs the llama.cpp build with the DeBERTa-v2 patch (patches/llama.cpp-deberta-v2.diff
# in the laya-finetune repo); the stock libs in LLAMA_SRC cannot load it. Q8_0 because Go's linker caps embedded data at 2 GB.
# It is the default model (LAYA_MODEL), embedded with its own libs (assets/llama-deberta).
GLINER_SRC ?= $(HOME)/Projects/laya-finetune/out/gliner2-decide-gguf
# Host build of that patched llama.cpp, used only by `make dev-gliner` (assets/llama-deberta comes from `make llama-deberta`)
# (cmake -B build-portable -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF
# -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_RPATH='$$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH='$$ORIGIN').
# CPU variants are picked at run time, like the stock libs. $$ORIGIN matters: a default build points RUNPATH at its own
# build dir, so the libs load here but not on any other machine ("libggml-base.so.0: cannot open shared object file").
GLINER_LLAMA ?= $(HOME)/Projects/laya-finetune/tools/llama.cpp/build-portable/bin
LAYA_API_KEYS ?= devkey

export LAYA_API_KEYS
# AVX2 dot/axpy for the decision head (laya/dot_simd.go): ~40% faster head. Without it
# the build silently uses the scalar path; `GOEXPERIMENT= make build` forces that.
GOEXPERIMENT ?= simd
export GOEXPERIMENT

# Docker Hub: make docker-push DOCKERHUB_USER=me [VERSION=1.0.0]   (run `docker login` first)
DOCKERHUB_USER ?= djanonurer
VERSION ?= latest
IMAGE = $(DOCKERHUB_USER)/laya-server
GPU_TAG = $(if $(filter latest,$(VERSION)),cuda,$(VERSION)-cuda)

.PHONY: assets assets-stock assets-guard assets-gliner llama-deberta dev-gliner dev-qwen3guard docker-build-stock build run dev test push push-gpu docker-build docker-push docker-build-gpu docker-push-gpu check-user
# Copy the real (non-symlink) libs and both models into assets/ for go:embed.
assets: assets-stock assets-guard assets-gliner
assets-stock:
	mkdir -p assets/llama assets/laya
	cp $(LLAMA_SRC)/libggml.so.0.24.0 $(LLAMA_SRC)/libggml-base.so.0.24.0 $(LLAMA_SRC)/libllama.so.0.4.1 $(LLAMA_SRC)/libggml-cpu-*.so assets/llama/
	cp $(MODEL_SRC)/tokenizer.json $(MODEL_SRC)/laya-multilingual-head.safetensors $(MODEL_SRC)/laya-multilingual-F16.gguf assets/laya/
# Copy the guard model into assets/laya-guard (embedded next to the stock model; select with LAYA_MODEL).
assets-guard:
	mkdir -p assets/laya-guard
	cp $(GUARD_SRC)/*.gguf $(GUARD_SRC)/laya-multilingual-head.safetensors $(GUARD_SRC)/tokenizer.json assets/laya-guard/
	chmod 644 assets/laya-guard/*
# Copy GLiNER2.5-Decide (Q8_0 encoder, head, tokenizer; the default model) into assets/gliner2-decide, and build the
# patched llama.cpp it needs into assets/llama-deberta (see llama-deberta).
assets-gliner: llama-deberta
	mkdir -p assets/gliner2-decide
	cp $(GLINER_SRC)/gliner2-decide-encoder-Q8_0.gguf $(GLINER_SRC)/gliner2-decide-head.safetensors $(GLINER_SRC)/tokenizer.json assets/gliner2-decide/
	chmod 644 assets/gliner2-decide/*
# Patched llama.cpp (patches/llama.cpp-deberta-v2.diff) built in the Dockerfile's Debian bookworm stage, so the libs run
# on glibc >= 2.36 (a build on this host would need the host's newer glibc) and on any x86-64 CPU (variants picked at run time).
llama-deberta:
	rm -rf assets/llama-deberta
	docker build --target llama-deberta-libs --output type=local,dest=assets/llama-deberta .
	chmod 644 assets/llama-deberta/*
	@! readelf -d assets/llama-deberta/*.so* | grep -E 'R(UN)?PATH' | grep -v '\[\$$ORIGIN\]' || { echo "assets/llama-deberta: a lib has an absolute RUNPATH"; exit 1; }
build:
	go build -trimpath -ldflags="-s -w" -o laya-server .
run:
	go run .
# Rebuild and restart on .go changes (see .air.toml). LAYA_DIR/LAYA_LLAMA_LIB skip the asset extraction.
dev:
	go tool air
# `make dev` with GLiNER2.5-Decide: the patched llama.cpp libs replace the embedded ones.
dev-gliner:
	LAYA_MODEL=gliner2-decide LAYA_LLAMA_LIB=$(GLINER_LLAMA) go tool air
# `make dev` with Qwen3Guard-Gen-0.6B: too big to embed, so the server downloads it on first start (~0.8 GB, kept in the cache dir).
dev-qwen3guard:
	LAYA_MODEL=qwen3guard go tool air
test:
	go vet ./... && go test -race ./...
	GOEXPERIMENT= go test ./laya  # the scalar fallback must stay correct too

# make push [VERSION=1.0.0] [DOCKERHUB_USER=me]: build the CPU image and push it (run `docker login` first).
push: docker-push
push-gpu: docker-push-gpu

docker-build:
	docker build -t $(IMAGE):$(VERSION) .
docker-push: docker-build
	docker push $(IMAGE):$(VERSION)
# Same image contents, but runs the stock multilingual model by default (tag :stock).
docker-build-stock:
	docker build --build-arg DEFAULT_MODEL=laya -t $(IMAGE):stock .
check-user:
	@test -n "$(DOCKERHUB_USER)" || { echo "set DOCKERHUB_USER=<your Docker Hub username>"; exit 1; }

# CUDA image, tagged :cuda (or <version>-cuda). Slow: compiles llama.cpp with CUDA (needs `make assets` for the model).
docker-build-gpu: check-user
	docker build -f Dockerfile.gpu -t $(IMAGE):$(GPU_TAG) .
docker-push-gpu: docker-build-gpu
	docker push $(IMAGE):$(GPU_TAG)
