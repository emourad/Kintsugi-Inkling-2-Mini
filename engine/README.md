# The engine

The engine is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp): the same engine as our Kintsugi-27B
release, with Inkling 2 Mini support added. Only this engine can load the Kintsugi for Inkling 2 Mini model file. It
adds:

- **Trellis-coded experts**: the Escha trellis weights (EXL3 trellis format, 2- and 3-bit) for the MoE experts: decode
  and small batches run on the exllamav3 GEMV kernel (MIT, turboderp), and prompt reads use a tensor-core path on
  exactly decoded fp16 weights (the exact read).
- **Inkling 2 Mini** (arch `inkling2`, Unsloth's loop layout): the looped model (layers 6-17 run three times, each pass with its own K/V), the
  relative-position bias, the short convolutions, the attention sinks and the Inkling chat parser (tool calls and
  reasoning). Ported from Unsloth's llama.cpp mix `b12588-mix-c0c3de8` (MIT); see [CREDITS.md](../CREDITS.md) for the
  pull requests and their authors.
- **Quantized K/V in the banded attention**: the global attention reads a q4_0 K/V cache (the default; q8_0 on request) with the band bias
  and the per-query sinks, for decode and for prompt reads. Other builds read only f16 / bf16 / f32 K/V there.
- **A tiered K/V cache** (`--kv-vram-cells N`): the first N cells of each global K/V cache are in VRAM and the other
  cells are in pinned system RAM, in one CUDA address range. This is how the 262,144-token context fits a 12 GB card.
- **Batch-invariant kernels** (`GGML_CUDA_BATCH_INVARIANT=1`): the output for a token does not change with the batch
  it runs in.
- **Saved chats**: slot saves in the background, and delta saves (each save holds only the new part).
- **A spill guard**: a log line when the server's GPU memory goes past the card's usable limit.
- **A repetition guard** for agent loops (on by default for Inkling 2 Mini): see below.
- **Two slots (opt-in)**: Inkling 2 Mini can run with `-np 2 --kv-unified` (two requests at the same time, one shared
  window). The bundle launchers turn it on with `INKLING_SLOTS=2`. See "Known limits".

## Repetition guard

Sometimes the model calls the same tool again and again with nearly the same arguments and gets short results back
(for example `echo end1`, `echo end2`, ...). The server checks the end of each chat request for this: the last 6
assistant turns each made one call to the same tool, each call's arguments are nearly the same as the call before
(digits are ignored), and each result has 200 characters or fewer. Then the server adds one short note as the newest
message of that request only: "You have made nearly the same <tool> call <N> times in a row. Try a different step or
give your answer." The note is not in the client's own history. Each time the guard acts, the server log gets one WARN
line at the normal log level: `repetition guard (<endpoint>): N calls to '<tool>' in a row (rule 1, ...)`. The check
covers the chat, Anthropic messages and Responses endpoints.

- On by default for Inkling 2 Mini, off for other models. To turn it off: `--no-repeat-guard` (in the bundles:
  `INKLING_EXTRA=--no-repeat-guard`) or the environment variable `LLAMA_ARG_REPEAT_GUARD=0`. `--repeat-guard` turns it
  on for any model.
- Measured effect: at two stuck points it broke the loop in 6 and 7 of 10 tries, against 0 of 10 without it. When it
  does not act, the output does not change (50 of 50 test prompts token-identical with the guard on and off).
- After the guard acts, the next request reads about 4,000 tokens of the prompt again, one time.
- `--help` also names a second rule (8 or more calls to one tool); the server does not use that rule in this release.

## Source

The source of each release build is a release asset. Licence: MIT, the same as llama.cpp (`LICENSE` in the tar).

The engine source archive is the same tree as Kintsugi-27B's engine (its README names that project); this release
builds it at commit `a20cc419b` (build 10944) with the Inkling port and the pinned web UI. Both release engines (Windows and Linux)
are built from this exact archive, so you can rebuild it and compare.

| Engine | Bundles | Source asset | sha256 |
|---|---|---|---|
| `a20cc419b` | v1.0 (Windows and Linux) | `Kintsugi-Inkling-2-Mini-v1.0-engine-src-a20cc419b.tar.gz` (40,078,993 bytes) | `d0d934f5c8f6763477f4d8612000e6770e27b6260b426aa876460725319c6663` |

## Build

The release builds leave out the Mirai S kernels that are also in this source (`GGML_CUDA_MIRAI=OFF`) and build the web
UI from the `tools/ui/dist` folder of the source archive (no download). The configure lines of the v1.0 engines:

Windows (MSVC 19.44, CUDA 13):

```
cmake -S <source> -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75-real;86-real;89-real" -DGGML_CUDA_MIRAI=OFF -DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=OFF -DGGML_CUDA_GRAPHS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF -DLLAMA_BUILD_COMMIT=a20cc419b -DLLAMA_BUILD_NUMBER=10944
cmake --build <build> --target llama-server test-backend-ops llama-perplexity
```

Linux (Ubuntu 22.04, GCC 11.4, CUDA 12.8):

```
cmake -S . -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75-real;86-real;89-real" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CUDA_HOST_COMPILER=g++ -DGGML_CUDA_MIRAI=OFF -DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=ON -DGGML_CUDA_GRAPHS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF -DCMAKE_BUILD_RPATH='$ORIGIN' -DCMAKE_INSTALL_RPATH='$ORIGIN' -DGGML_CUDA_NCCL=OFF -DLLAMA_BUILD_COMMIT=a20cc419b -DLLAMA_BUILD_NUMBER=10944
cmake --build <build> --target llama-server llama-perplexity test-backend-ops
```

- **Linux:** `-DGGML_CUDA_NCCL=OFF` matters: with NCCL on, the engine needs `libnccl.so.2`, which a machine with only
  the NVIDIA driver does not have. The bundle also carries `libgomp.so.1` (see its NOTICE). The `tools/ui/dist` folder
  of the source archive is used before any download of a prebuilt UI.
- **One GPU family only:** set `CMAKE_CUDA_ARCHITECTURES` to it, for example `86-real` (RTX 30) or `89-real`
  (RTX 40). This release supports RTX 30 and 40 cards; the build also holds `75-real` code, which is not supported
  for this model yet.
- **CUDA toolkit:** 12.2 or newer. The Windows bundle uses CUDA 13; the Linux bundle uses CUDA 12.8.

## Known limits

- On Windows, OpenCode runs shell commands in PowerShell; the model sometimes writes bash syntax there and can get stuck repeating trivial commands. The repetition guard (above) adds a note in that case, but it does not always break the loop; then stop the turn and ask the model to use the edit tool.
- Inkling 2 Mini always runs with flash attention. With `-fa off`, the engine turns it on and writes a WARN line at the
  normal log level ("needs flash attention: flash_attn disabled -> on").
- One slot by default: with no `-np`, the server takes 1 slot. `-np 2 --kv-unified` (the bundles: `INKLING_SLOTS=2`)
  runs two slots that share one window. Limits of two slots: fewer K/V cells fit in VRAM (on an idle RTX 4070 with
  Windows 228,864 in place of 250,112, because each slot keeps its own sliding-window caches); saved conversations
  (the layer) are off; a long answer at temperature 0 is not bit-equal to the same request with one slot (where two
  tokens have almost the same probability, the batch layout can change which one comes first); and two slots do not
  make one user faster: with two long prompts at the same time, the total speed was about 10 % higher than one slot,
  and each slot was slower.
- K/V cache types other than f16, q8_0 and q4_0 (for example q5_0) refuse at start with a clear message.

## NVIDIA driver

- **Windows bundle:** it carries the CUDA 13 runtime, which needs driver 580 or newer. The launcher stops with a
  message on an older driver.
- **Linux bundle:** it carries the CUDA 12.8 runtime and works with driver 570 or newer. If the driver refuses the
  default type of pinned host memory (570 family), the engine uses the host NUMA type and writes one WARN line. If the
  driver refuses both, the load stops with "update the NVIDIA driver to 580 or newer". Test switch:
  `GGML_CUDA_HOSTMEM_FORCE=host|numa|pinned`.

## Where it comes from

- **llama.cpp** by the ggml authors and all its contributors (MIT).
- **Unsloth's llama.cpp mix** (MIT): the Inkling 2 Mini code, the banded flash-attention kernel and the parser we port.
- **Project Maya** ([maya-inkling-mini](https://github.com/alesha-pro/maya-inkling-mini), MIT; Peasant Smith and the
  Project Maya contributors, Niko1221 and the Strata contributors, Alexey Fateev, cobra91): the launch-overhead
  measurement and the idea to fuse the small glue launches; our fused ops are our own kernels and copy no Maya code.
  Its licence is `licenses/LICENSE-project-maya` (also in this folder).
- **PrismML's fork** ([PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp), MIT) is the direct base.
- **turboderp (exllamav3)** and **Ajay9o9 (llama.cpp-escha)**: kernels in the engine base (they run the coded experts).

The full list is in [CREDITS.md](../CREDITS.md).
