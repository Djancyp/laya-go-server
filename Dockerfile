# Run `make assets` first (assets/ is not in git). Verified: builds, runs non-root, /readyz in ~4s.
FROM golang:1.27 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOEXPERIMENT=simd go build -trimpath -ldflags="-s -w" -o /laya-server .

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -u 10001 laya && mkdir /var/cache/laya && chown laya /var/cache/laya
COPY --from=build /laya-server /usr/local/bin/laya-server
USER laya
# Libs and model are embedded; they are extracted to LAYA_CACHE_DIR on start (mount a volume to keep it).
# Probes: GET /healthz (liveness), GET /readyz (model loaded).
# Which embedded model runs by default: laya-guard (prompt-injection guard) or laya (stock multilingual).
# Override at run time with -e LAYA_MODEL=..., or at build time: --build-arg DEFAULT_MODEL=laya.
ARG DEFAULT_MODEL=laya-guard
ENV LAYA_MODEL=$DEFAULT_MODEL LAYA_CACHE_DIR=/var/cache/laya LAYA_ADDR=:8080
EXPOSE 8080
ENTRYPOINT ["laya-server"]
