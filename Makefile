# Source of the files embedded into the binary (see `make assets`).
LLAMA_SRC ?= $(HOME)/.kronk/libraries/linux/amd64/cpu
MODEL_SRC ?= $(HOME)/Documents/codding/experiment/models/laya
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

.PHONY: assets build run dev test docker-build docker-push docker-build-gpu docker-push-gpu check-user
# Copy the real (non-symlink) libs and the model files into assets/ for go:embed.
assets:
	mkdir -p assets/llama assets/laya
	cp $(LLAMA_SRC)/libggml.so.0.24.0 $(LLAMA_SRC)/libggml-base.so.0.24.0 $(LLAMA_SRC)/libllama.so.0.4.1 $(LLAMA_SRC)/libggml-cpu-*.so assets/llama/
	cp $(MODEL_SRC)/tokenizer.json $(MODEL_SRC)/laya-multilingual-head.safetensors $(MODEL_SRC)/laya-multilingual-F16.gguf assets/laya/
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

docker-build: 
	docker build -t $(IMAGE):$(VERSION) .
docker-push: docker-build
	docker push $(IMAGE):$(VERSION)
check-user:
	@test -n "$(DOCKERHUB_USER)" || { echo "set DOCKERHUB_USER=<your Docker Hub username>"; exit 1; }

# CUDA image, tagged :cuda (or <version>-cuda). Slow: compiles llama.cpp with CUDA (needs `make assets` for the model).
docker-build-gpu: check-user
	docker build -f Dockerfile.gpu -t $(IMAGE):$(GPU_TAG) .
docker-push-gpu: docker-build-gpu
	docker push $(IMAGE):$(GPU_TAG)
