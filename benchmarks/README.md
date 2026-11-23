# Benchmarks: the numbers on the card

All results below are for `Kintsugi-Inkling-2-Mini-12G.gguf` (sha256 `7d93d5e8e61f6694776ec8b488751787a7dd3cb745d4f9eaee9f0b78b0e60cec`) on our engine, measured on the
exact v1.0 bundle at its shipped settings, unless a row says otherwise.

**Rig:** one RTX 4070 12 GB, Windows 11, display on the integrated GPU, NVIDIA driver 580+ family, memory clock
11,751 MHz under load (checked for each speed row).

**Launcher settings (the bundle's defaults):** window 262,144 tokens; the 8 global K/V caches q4_0, 250,112 cells in
VRAM; the 40 sliding-window caches f16; micro-batch 1,024; the exact prompt read; thinking effort high (0.9); temperature 1.0,
top-p 1.0, top-k 0, min-p 0; no drafting.

| Card number | Test | Settings | Date | Result |
|---|---|---|---|---|
| Fidelity to the official weights | Mean KL of the next-token distribution (top-512 + tail, the challenge's arithmetic) against Thinking Machines' official weights (revision `ed40e7da`) in our exact PyTorch forward. Golden set: 298 public documents (228 text, 70 chat) of the Qwen3.8-27B quant challenge corpus (StillDeadcode, CC BY 4.0; documents 0, 2, 4, ...; not used for the calibration), re-rendered to Inkling's format, cut to 32,768 tokens: 595,802 tokens, 451,585 scored (every text token; the assistant tokens of the chats) | Our engine (build `4c23a71f8`), f16 K/V | 2026-11-21 | **mean KL 0.0310**, same top token **93.62 %**, perplexity 4.105 (official 4.032: ratio 1.018); see note 1 |
| The same test, Unsloth's file `Inkling-2-Mini-UD-Q2_K_XL.gguf` (Unsloth UD-Q2_K_XL, 9,138,715,328 bytes) in the same engine | As above | As above | 2026-11-21 | mean KL 0.0426, same top token 92.70 %, perplexity 4.140. Ours (0.0310): **27.3 % lower KL** (95 % paired bootstrap interval of the difference [-0.0133, -0.0102]), lower on 281 of 298 documents and in all 14 categories; with the release-default K/V (q4_0 global + f16 window caches) in both, on a 10-document subset of the golden set (13,772 scored tokens): **mean KL 0.0400 vs 0.0553, 27.7 % lower** (ours with q4_0 against ours with f16: +0.0009). One limit: on the non-scored chat tokens (user and tool text) past 1,000 tokens, our KL is higher (0.76 vs 0.61); the needle and code-copy tests were perfect for both files |
| Context on 12 GB: 262,144 | The launcher's default window, 262,144 tokens: prompts of 245,000 tokens read and answered (the recall row), with VRAM under the card's line and no spill (the memory rows) | Launcher defaults. For reference: with f16 K/V all in VRAM (the usual path for this model), about 65,000 tokens fit on the same card | 2026-11-21 | **262,144-token window; recall shown to 245,000 tokens** |
| The 4-bit long-context cache (the default) | Our q4_0 K/V against our f16 K/V with this model file: llama-perplexity, -c 32,768, -ub 1,024, 4 chunks of WikiText-2 | f16 window caches in both | 2026-11-22 | **KL 0.004552 (line 0.005: PASS), same top token 97.07 %, perplexity ratio 1.0014** |
| The 8-bit cache (`INKLING_KV=q8_0`) | The same check, q8_0 against f16 | The same run | 2026-11-22 | KL 0.001861, same top token 98.20 % |
| The tiered cache | q8_0 with all cells in VRAM against q8_0 with most of a 32k window in system RAM | q8_0, micro-batch 1,024 (the tier code is the same for q4_0) | 2026-11-21 | **KL 0.000000 (line 1e-5), same top token 99.997 %: PASS** |
| Recall at 64k / 117k / 245k | 24 needle questions and 20 exact code copies at each depth (the frozen bench scripts) | Thinking off (effort none), temperature 0; q4_0 global + f16 window caches (the release default), window 262,144, 250,112 cells in VRAM, layer off; release engine `a20cc419b` | 2026-11-22 | **64k, 117k and 245k: 24/24 needles + 20/20 copies (20/20 with the correct lead) at each depth.** VRAM peak 11,459 / 11,457 / 11,457 MiB, 0 CUDA errors, 0 spill. At 245,000 tokens the whole prompt is in VRAM (250,112 cells) |
| Memory | VRAM trace over identity (50 prompts), 40 varied prompts and one 245,279-token read, after the launcher's two warm-up requests (the first prompts add about 210 MiB once, then VRAM stays flat; the launcher keeps 300 MiB for this growth on top of its margin) | Launcher defaults (q4_0, layer on, micro-batch 1,024, window 262,144), release engine `a20cc419b`; tested with a simulated desktop load of 500 / 900 MiB (VRAM held by a separate process); not yet with a display on the same card | 2026-11-22 | **No display: 250,112 cells in VRAM, peak 11,687 MiB. 500 MiB simulated desktop load: 185,088 cells, peak 11,684 MiB. 900 MiB: 133,120 cells, peak 11,692 MiB** (line 11,935). After the warm-up: 11,479 / 11,476 / 11,484 MiB. Shared-memory rise 0, 0 spill, 0 out-of-memory or CUDA errors; 40 of 40 varied prompts answered. The 245k read: prompt read 1,178 / 1,176 / 1,179 tok/s, decode 30.6 / 22.0 / 15.5 tok/s (32 tokens; less of the context is in VRAM with a desktop load) |
| Memory, Linux | The same trace on Linux: identity 50 / 50, 40 varied prompts with 0 failed, and a 245,279-token read answered | Ubuntu 24.04, driver 610.57.04, only the NVIDIA driver (no CUDA toolkit, no NCCL); RTX 4070; earlier build `ad657a606`, launcher defaults (q4_0, layer on), no display | 2026-11-22 | **236,288 cells in VRAM** (driver reserve 380 MiB, limit 11,894), VRAM 11,395 MiB after the warm-up, **peak 11,603 MiB** (line 11,830); growth after the warm-up 208 MiB (the launcher keeps 300); 0 out-of-memory, CUDA error or spill lines |
| Decode / prompt read | quick_tps: a fresh server for each depth, 2 runs, the mean; layer off | Launcher defaults; Unsloth's build at the same depths | 2026-11-22 | see the table below |
| Agent tasks vs the official weights | AppWorld test_normal, seeds 102, 103, 104 and 105: 672 paired task-runs, paired exact test on the discordant tasks | Effort medium, window 65,536. Ours: this model file with the release default (q4_0 global + f16 window caches) on our earlier build `ad657a606`. The official weights: Thinking Machines' release (NVFP4 / MXFP8, revision `ed40e7da`), exact, with f16 K/V, in the same engine with the same flags | 2026-11-22 | **595 / 672 vs 592 / 672 (p = 0.78): level with the official weights.** The 8-bit cache (`INKLING_KV=q8_0`) gives 596 / 672 (595 vs 596: p = 1.00). With the full 262,144 window (seeds 102 + 103): 296 / 336 vs 298 / 336 with the 8-bit cache at 65,536 (p = 0.85) |
| Agent and code tasks vs Unsloth's setup | AppWorld test_normal (168 tasks x 2 seeds = 336 task-runs, paired exact test on the discordant tasks) and HumanEval (164, pass@1) | Effort medium, window 65,536. Ours: this model file on our engine, with the 8-bit cache (`INKLING_KV=q8_0`). Unsloth's file (UD-Q2_K_XL) on Unsloth's llama.cpp build | 2026-11-22 | **AppWorld, seeds 102 + 103: 298 / 336 vs 283 / 336 (p = 0.0135); seeds 104 + 105: 298 / 336 vs 281-283 / 336 (p = 0.0033-0.0081): better. HumanEval 158 / 164 vs 160 / 164: level (p = 0.63).** On seeds 104 + 105, 2 of Unsloth's task-runs were stopped unfinished (after 4 h 47 min and 7 h 17 min, each answering at the 24,576-token output cap); the result holds for any outcome of those 2. This compares the two shipped setups at the same effort and window: our file in our engine (q8_0 K/V, reasoning budget 20,480) against Unsloth's file in Unsloth's build (f16 K/V, no budget). It does not separate the weight file from the engine. |
| Works with: Claude Code, Codex, Pi, OpenCode | 6 small coding tasks in a small Python repository (see [harnesses/](../harnesses/README.md)) | Earlier build `ad657a606`, launcher defaults (q4_0), through the saved-chat layer, Windows 11 | 2026-11-22 | **5 / 6 in each of the four.** 0 malformed calls, 0 loops, 0 server errors, 0 leaked tags. The missed task is the same in all four (a model answer). With the 8-bit cache, OpenCode once looped on a Windows shell task (model behaviour; see the known limits). Again with the repetition guard on (the release code before the two-slot change): 5 / 6 in each of the four, 0 loops, 0 server errors; the guard did not act |
| Repetition guard | The recorded requests of the OpenCode loop, replayed 10 times at temperature 1.0 at each of 3 points, guard on and off: does the next call repeat the trivial command? Also 908 recorded agent requests of earlier harness runs | The release guard code (rule: the last 6 assistant turns, one call each to the same tool, nearly the same arguments, results of 200 characters or fewer) | 2026-11-22 | **At two stuck points it broke the loop in 6 and 7 of 10 tries, against 0 of 10 without it.** At the third point (before the loop is long enough) the guard does not act. It acted only inside the known loop (0 of the other recorded requests). With the guard off, the output at temperature 0 is bit-equal to the engine without the guard. After the guard acts, the next request reads about 4,000 tokens of the prompt again, one time |
| Two slots (opt-in, `INKLING_SLOTS=2`) | Two prompts at the same time (25,745 and 28,847 tokens) through the launcher, VRAM and shared-memory trace; short answers of two different prompts against one slot | The two-slot code of the release engine, Windows 11, layer off, q4_0 | 2026-11-22 | **228,864 cells in VRAM (one slot: 250,112), peak 11,477 MiB (line 11,999), 0 spill.** Decode 38.0 and 27.1 tok/s at the same time (one slot, the same prompts one after the other: 59.0 and 58.5). Short answers token-equal to one slot, and no text of one slot in the other. Long answers at temperature 0 can differ from one slot at a near tie (the second-choice token) |
| Token check | 6 long generated samples (11 rounds), every output token checked for malformed tokens, with logprobs | Release engine `a20cc419b`, the staged launcher, layer off | 2026-11-22 | **6 / 6 samples, 72,406 tokens, 0 malformed-token findings**, 0 HTTP errors, every page renders |

## Speed

| Depth | Ours (release engine `a20cc419b`), no display: prompt read / decode | Ours, 500 MiB desktop load (earlier build `ad657a606`): prompt read / decode | Unsloth's build (UD-Q2_K_XL, f16 K/V): prompt read / decode |
|---|---|---|---|
| 4k | 2,277 / 64.5 | - | 1,130 / 59.1 |
| 32k | 2,150 / 57.9 | 2,150 / 58.45 | 1,083 / 52.1 |
| 63k | - | - | 1,024 / 46.2 (its 12 GB maximum) |
| 117k | 1,610 / 42.4 | 1,610 / 42.65 | does not fit on 12 GB |
| 245k | 1,161 / 29.75 | 1,162 / 21.25 | does not fit on 12 GB |

All in tokens per second, RTX 4070, launcher defaults (q4_0 K/V, window 262,144), no drafter, plain server, a fresh
server for each depth, 2 runs. Prompt read: tokens per second of the first prompt on a fresh server, which reads the whole
context; mean of 2 runs. Decode: 400 tokens on each of 3 prompts, the mean of the 3, mean of 2 runs (as on the
Kintsugi-27B page). Ours: the release engine `a20cc419b` with no display, 2026-11-22 (the two 4k runs differ by 5 %,
62.9 and 66.1: the usual run-to-run noise of one request); the 500 MiB load on the earlier build `ad657a606` (the same
K/V code); depths 4,000 / 32,000 / 117,000 / 245,000. Unsloth's build b12588: its UD-Q2_K_XL file, f16 K/V, no drafter, window = depth + 4,096 (depths
4,096 / 32,768 / 63,488). **At 4k our decode is 1.09x Unsloth's build, at 32k 1.11x, and our prompt read is about 2x at
both.** At 245k the 4-bit cache decodes 29.75 tok/s, 3.4x the 8-bit cache of the same engine (8.7 tok/s on the earlier
build `ace85a829`, `INKLING_KV=q8_0`): more of the context fits in VRAM. Cells in VRAM: 250,112 with no display (a 245k prompt fits in VRAM), 185,088 with the 500 MiB load (past 185,088 tokens the K/V cache is in system RAM and streams over PCIe at each step, which is why 245k is slower there). Deeper rows are also slower because each new token attends to more of the context.

## The release engine `a20cc419b`

The release engine `a20cc419b` (build 10944) is `ad657a606` with three additions: the WARN line for `-fa off`, the
repetition guard and the opt-in two slots. With one slot (the default), its output is the same as the builds measured
above: on the release engine itself, its G1 quality check is identical (KL 0.003341, same top token 97.727 %), and
50 of 50 test prompts through the release launcher are token-identical to the build before the two-slot change. On 2026-11-22 the release engine itself, with the staged
launcher, gave the speed (no display), recall, memory and token-check rows above. The other rows keep the build they
were measured on.

## Notes

1. The fidelity gate of the model file had four lines. Three passed. The fourth, the per-document correlation of our
   engine with our Python simulation of the same file, is **FAIL as written: per-document engine-vs-Python correlation
   r = 0.919 against a 0.95 line; Unsloth's own file gives 0.912 by the same mechanism (tiny tool documents); on
   documents with 500+ scored rows r = 0.997.** The mean KL of the engine and of the simulation agree within 0.00004.
