//go:build cuda

package main

import "embed"

// CUDA build (Dockerfile.gpu): llama.cpp compiled with CUDA, embedded instead of the CPU libs.
const libAssetsDir = "llama-cuda"

//go:embed assets/llama-cuda assets/laya assets/laya-guard
var assets embed.FS
