# The model file

| | |
|---|---|
| File | `Kintsugi-Inkling-2-Mini-12G.gguf` |
| Size | 9,067,638,560 bytes (9.07 GB) |
| SHA-256 | `7d93d5e8e61f6694776ec8b488751787a7dd3cb745d4f9eaee9f0b78b0e60cec` |
| Download | [huggingface.co/CaryPalmer/Kintsugi-Inkling-2-Mini](https://huggingface.co/CaryPalmer/Kintsugi-Inkling-2-Mini) |
| Base model | [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (26.4B, looped mixture of experts) |
| Licence | Apache-2.0, plus the Thinking Machines Model Acceptable Use Policy |

## Format

- **Experts in EXL3 trellis code**: each of the 128 experts of each of the 24 layers (gate and up fused, and down) is
  coded at 2 or 3 bits per weight, in the Escha W2 format (EXL3 trellis code with Hadamard input and output scales):
  2.444 bits per weight on average over the experts. Map M2: the down projections at 3 bits in all 24 layers, the
  fused gate and up projections at 3 bits in the 4 global layers (5, 11, 17, 23) and at 2 bits in the other 20.
- **Attention projections**: Q8_0. **Loop controllers, sinks, convolution projections**: F16. **Router and norms**:
  F32 (the routing stays exact).
- **Output head**: Q8_0. **Token embedding**: Q4_0.
- Text only: the image part of the model is not in the file.

Every tensor that is not a coded expert comes from Thinking Machines' official release (revision `ed40e7da`). The
GGUF metadata, the tokenizer and the chat template are as in Unsloth's GGUF of the model (Apache-2.0).

## The model in short

- 24 weight layers, hidden size 2,304, vocabulary 201,024 (padded).
- A mixture of experts in every layer: 128 experts, 8 for each token, no shared expert.
- **The loop**: layers 6-17 run three times with the same weights. So each token goes through 48 layer applications,
  and each application has its own K/V cache: 8 global (full-context) attention caches and 40 sliding-window caches
  (window 1,024).
- Global attention: 8 K/V heads of 128 values. At f16 that is 32 KiB of K/V for each token; at q4_0 (this release),
  9 KiB. 262,144 tokens at q4_0 = 2.25 GiB, of which the part past the VRAM line is in system RAM.

## How it was made

- **Hessians**: captured on our own forward pass of the official weights, with calibration text from the public
  corpus of the Qwen3.8-27B quant challenge (StillDeadcode, CC BY 4.0), re-rendered with Inkling's chat template. Every
  expert saw every calibration token during the capture (not only the routed ones), and the three passes of the loop
  add into one Hessian for each weight.
- **Fits**: exllamav3's `quantize_exl3` (turboderp, MIT), 2-bit and 3-bit for every expert matrix; the map chooses one
  of the two for each layer.
- **Measured** against the official weights on a golden set from the same public corpus (documents not used for the
  calibration): mean KL 0.0310, same top token 93.62 % (see
  [benchmarks/](../benchmarks/README.md)).

## Chat template

The template inside the .gguf is Thinking Machines' template (as Unsloth's GGUF carries it). It takes the thinking
effort as `reasoning_effort` (none, minimal, low, medium, high, xhigh, max, or a number from 0.0 to 0.99; default
0.9). It needs no fix for coding harnesses: it accepts a system message at any position (Claude Code sends one after
the first turn), and the server maps "developer" messages to "system" before the template (Codex sends two).

## What can load it

Only our engine (see [engine/](../engine/README.md)), which is in the bundle. Other llama.cpp builds, Ollama and LM
Studio cannot load it: they do not have the trellis op for the experts.
