# Source of the files embedded into the binary (see `make assets`).
LLAMA_SRC ?= $(HOME)/.kronk/libraries/linux/amd64/cpu
MODEL_SRC ?= $(HOME)/Documents/codding/experiment/models/laya
# Calibrated prompt-injection guard model (output of scripts/v2/to_gguf.py in the laya-finetune repo).
GUARD_SRC ?= $(HOME)/Documents/codding/experiment/models/laya-guard
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

.PHONY: assets assets-stock assets-guard docker-build-stock build run dev test push push-gpu docker-build docker-push docker-build-gpu docker-push-gpu check-user
# Copy the real (non-symlink) libs and both models into assets/ for go:embed.
assets: assets-stock assets-guard
assets-stock:
	mkdir -p assets/llama assets/laya
	cp $(LLAMA_SRC)/libggml.so.0.24.0 $(LLAMA_SRC)/libggml-base.so.0.24.0 $(LLAMA_SRC)/libllama.so.0.4.1 $(LLAMA_SRC)/libggml-cpu-*.so assets/llama/
	cp $(MODEL_SRC)/tokenizer.json $(MODEL_SRC)/laya-multilingual-head.safetensors $(MODEL_SRC)/laya-multilingual-F16.gguf assets/laya/
# Copy the guard model (the default) into assets/laya-guard (embedded next to the stock model; select with LAYA_MODEL).
assets-guard:
	mkdir -p assets/laya-guard
	cp $(GUARD_SRC)/*.gguf $(GUARD_SRC)/laya-multilingual-head.safetensors $(GUARD_SRC)/tokenizer.json assets/laya-guard/
	chmod 644 assets/laya-guard/*
build:
	go build -trimpath -ldflags="-s -w" -o laya-server .
run:
	go run .
# Rebuild and restart on .go changes (see .air.toml). LAYA_DIR/LAYA_LLAMA_LIB skip the asset extraction.
dev:
	go tool air
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
