# Kintsugi for Inkling 2 Mini bundle: changelog

The model file is `Kintsugi-Inkling-2-Mini-12G.gguf` (9,067,638,560 bytes, sha256
`7d93d5e8e61f6694776ec8b488751787a7dd3cb745d4f9eaee9f0b78b0e60cec`, from [Hugging
Face](https://huggingface.co/CaryPalmer/Kintsugi-Inkling-2-Mini)). It is the same for every version below: when a new
bundle comes out, download only the new bundle. Each release on the [Releases
page](https://github.com/professorpalmer/Kintsugi-Inkling-2-Mini/releases) also has the engine source.

## Planned for a later version

- RTX 20 (Turing) cards.

## v1.0 (2026-11-23)

- First release: Thinking Machines' Inkling 2 Mini in our own model file (every expert in EXL3 trellis code at 2 or 3
  bits), on our llama.cpp fork, with a 262,144-token window on one 12 GB card.
- The 8 global attention caches (the long context) use 4-bit (q4_0) K/V on a tiered cache: the first cells are in
  VRAM, the rest are in pinned system RAM. On an RTX 4070 with no display on it, 250,112 of the 262,144 cells are in VRAM.
  `INKLING_KV=q8_0` gives the 8-bit cache (fewer cells in VRAM).
  The 40 sliding-window caches are f16 and stay in VRAM.
- The launcher reads the memory that the NVIDIA driver keeps for itself, keeps the server under the rest, checks the
  driver (Windows: 580 or newer; Linux: 570 or newer) and stops with a clear message when the engine can not use the
  GPU.
- Thinking effort: the model's own default (`high`, 0.9); `INKLING_EFFORT` changes it. Sampling: the model maker's
  settings (temperature 1.0, top-p 1.0, top-k 0, min-p 0).
- Saved chats through the layer (Python 3.10 or newer): a long chat continues after a restart without a full read.
- A repetition guard (on by default): when the model repeats nearly the same tool call with short results, the server
  adds one short note to that request only and writes one WARN line. At two stuck points it broke the loop in 6 and 7
  of 10 tries, against 0 of 10 without it. `LLAMA_ARG_REPEAT_GUARD=0` or `--no-repeat-guard` turns it off.
- Two slots (opt-in, `INKLING_SLOTS=2`): two requests at the same time in one shared window. Fewer cells in VRAM,
  saved chats off, long answers not bit-equal to one slot (see the README).
- `-fa off` writes a WARN line in the normal log (the engine always runs Inkling 2 Mini with flash attention).
- Tested with Claude Code, Codex, Pi and OpenCode (5 of 6 tasks each); no malformed tool calls, no loops and no
  server errors. Details: `HARNESSES.md`.
- Windows: `Kintsugi-Inkling-2-Mini-v1.0-win-x64.zip`, sha256 `85ddccfac0d4434a205b1eb6269f4505da93573ce8801569d46a33869f127c2a`.
- Linux: `Kintsugi-Inkling-2-Mini-v1.0-linux-x86_64.tar.gz`, sha256 `906ab5cde3964c85723243520b774143b6d2901cd75be8c64db6f1996f76a6d4`.
- Engine source: `Kintsugi-Inkling-2-Mini-v1.0-engine-src-a20cc419b.tar.gz` (engine `a20cc419b`, build 10944), sha256 `d0d934f5c8f6763477f4d8612000e6770e27b6260b426aa876460725319c6663`.
