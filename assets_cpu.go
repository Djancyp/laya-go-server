//go:build !cuda

package main

import "embed"

// libAssetsDir is where the embedded llama.cpp libraries are extracted, under the cache dir.
const libAssetsDir = "llama"

//go:embed assets/llama assets/laya assets/laya-guard
var assets embed.FS
