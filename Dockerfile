# Run `make assets` first (assets/ is not in git). Verified: builds, runs non-root, /readyz in ~4s.

# llama.cpp with the DeBERTa-v2 patch (patches/), which gliner2-decide needs. Built here, on the runtime's Debian, so
# the libs match its glibc (libs built on a newer distro fail with "version `GLIBC_2.43' not found"). Portable: all
# CPU variants, picked at run time; RUNPATH $ORIGIN so the libs find each other wherever they are extracted.
# `make llama-deberta` exports these libs to assets/llama-deberta for local builds.
FROM debian:bookworm AS llama-deberta
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ARG LLAMA_COMMIT=7677678503e921f98da87e77bde531752977848e
WORKDIR /llama.cpp
RUN git init -q . && git remote add origin https://github.com/ggml-org/llama.cpp \
    && git fetch -q --depth 1 origin ${LLAMA_COMMIT} && git checkout -q FETCH_HEAD
COPY patches/llama.cpp-deberta-v2.diff /tmp/
RUN git apply /tmp/llama.cpp-deberta-v2.diff
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF \
      -DCMAKE_BUILD_RPATH='$ORIGIN' -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
      -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
    && cmake --build build -j"$(nproc)" \
    && mkdir /out && cd build/bin && cp libggml.so.0.* libggml-base.so.0.* libllama.so.0.* libggml-cpu-*.so /out/

# Only the libs, for `docker build --target llama-deberta-libs --output assets/llama-deberta .`
FROM scratch AS llama-deberta-libs
COPY --from=llama-deberta /out/ /

FROM golang:1.27 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
# The patched libs come from the stage above, never from the host (whose glibc may be newer).
RUN rm -rf assets/llama-deberta
COPY --from=llama-deberta /out/ assets/llama-deberta/
RUN CGO_ENABLED=0 GOEXPERIMENT=simd go build -trimpath -ldflags="-s -w" -o /laya-server .

FROM debian:bookworm-slim
# ca-certificates: without it the image has no root CA pool at all, so every
# outbound HTTPS call fails "x509: certificate signed by unknown authority" —
# including POST /v1/policy's first-use download of its chat model from
# Hugging Face. update-ca-certificates runs automatically as part of install.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -u 10001 laya && mkdir /var/cache/laya && chown laya /var/cache/laya
COPY --from=build /laya-server /usr/local/bin/laya-server
USER laya
# Libs and model are embedded; they are extracted to LAYA_CACHE_DIR on start (mount a volume to keep it).
# Probes: GET /healthz (liveness), GET /readyz (model loaded).
# Which embedded model runs by default: gliner2-decide (runs on the patched llama.cpp in assets/llama-deberta),
# laya-guard (prompt-injection guard) or laya (stock multilingual).
# Override at run time with -e LAYA_MODEL=..., or at build time: --build-arg DEFAULT_MODEL=laya-guard.
# POST /v1/policy downloads its chat model (~0.6 GB) from Hugging Face on first use; keep /var/cache/laya on a volume.
ARG DEFAULT_MODEL=gliner2-decide
ENV LAYA_MODEL=$DEFAULT_MODEL LAYA_CACHE_DIR=/var/cache/laya LAYA_ADDR=:8080
EXPOSE 8080
ENTRYPOINT ["laya-server"]
