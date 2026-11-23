# Kintsugi for Inkling 2 Mini v1.0 (Windows, RTX 30 / 40, 12 GB)

![Kintsugi for Inkling 2 Mini](inkling-card.png)

Kintsugi for Inkling 2 Mini is Thinking Machines' Inkling 2 Mini (a 26.4B looped mixture-of-experts model) in our own
2-3 bit model file, built to run on one 12 GB NVIDIA card with a 262,144-token context. The engine keeps the
long-context cache at 4 bits (q4_0): on a free card up to 250k tokens stay in VRAM, and with a desktop on the card the
rest goes to system RAM, so a long chat works on 12 GB. Free to use, under
Apache-2.0. Found a problem or have numbers from your card? Please open a GitHub issue (template at the end); pull
requests are welcome.

**Download:** the Windows bundle (engine, launcher, saved-chat layer) is on the
[Releases page](https://github.com/professorpalmer/Kintsugi-Inkling-2-Mini/releases); the model file is on
[Hugging Face](https://huggingface.co/CaryPalmer/Kintsugi-Inkling-2-Mini).

## What you need

- An NVIDIA RTX 30 or 40 series card with 12 GB of VRAM or more: for example RTX 3060 12 GB, 3080 12 GB, 3080 Ti,
  3090, 4060 Ti 16 GB, 4070 and up, 4090. RTX 20 (Turing) is NOT supported; RTX 50 (Blackwell) not yet (see "Hardware").
  Cards with less than 12 GB can not hold the model. On 16-24 GB cards the launcher keeps more of the context in VRAM,
  so long-context speed is higher there.
- NVIDIA driver 580 or newer (the bundle carries its own CUDA 13 runtime).
- Windows 10 or 11, 64-bit. 32 GB of system RAM or more is best for very long chats.
- About 10 GB of disk for the model, plus room for saved chats (up to 20 GB, in `slots\`).
- Optional: Python 3.10 or newer, for saved chats (see below). No extra Python packages are necessary.

## Hardware

| Hardware | Status |
|---|---|
| RTX 40 (Ada) | Supported: every release check ran on an RTX 4070 (Windows and Linux). |
| RTX 30 (Ampere) | Supported: the release engine was checked on an RTX 3080 Ti (Linux): it starts with the launcher defaults, 24 / 24 needles at 32k, the identity prompts answer, top-1 agreement 97.8 %. Its mean KL is 7.5 % lower than on the RTX 4070; without one near-tie token in the 4070 reference run the two cards differ by 0.6 % (the same kernels; only the order of sums differs). |
| RTX 50 (Blackwell), H100 / H200 (Hopper) | Not yet: this engine has no code for them. Planned for v1.0.1. |
| RTX 20 (Turing) | Not supported: it fails our exact-output checks and runs 6-14x slower than an RTX 4070. |
| AMD Radeon | Not yet: after Kintsugi-27B's AMD release (test builds exist). |
| Apple Silicon | Not yet: needs its own port; not scheduled. |

You need 12 GB of VRAM or more, and an NVIDIA driver 580 or newer (Windows) or 570 or newer (Linux). 32 GB of system
RAM or more is best for very long chats.

## Download and start

1. Download the bundle zip from the Releases page and unzip it to a folder, for example `C:\Kintsugi-Inkling`.
2. Download `Kintsugi-Inkling-2-Mini-12G.gguf` (9.07 GB) from
   https://huggingface.co/CaryPalmer/Kintsugi-Inkling-2-Mini and put it in the `models` folder. Its SHA-256 is
   `7d93d5e8e61f6694776ec8b488751787a7dd3cb745d4f9eaee9f0b78b0e60cec`. Check it in PowerShell with
   `Get-FileHash models\Kintsugi-Inkling-2-Mini-12G.gguf`.
3. Open PowerShell in the folder and start the server:

   ```
   powershell -ExecutionPolicy Bypass -File start-server.ps1 -Wait
   ```

   The first start takes about a minute. Wait for the line `ready  http://127.0.0.1:8080/v1`.
4. Chat (see "How to chat" below). The server listens only on this computer.
5. To stop it: `powershell -ExecutionPolicy Bypass -File stop.ps1`.

## How to chat

- **In the browser:** open http://127.0.0.1:8080/ (the chat page that comes with the server).
  The web page needs a browser (it is served gzip-compressed); for scripts use the API at /v1.
- **From an app:** point any OpenAI-compatible app at `http://127.0.0.1:8080/v1` (model name: anything; no key).
- **Agent harnesses** (Claude Code, Codex, Pi, OpenCode): see `HARNESSES.md` for the settings we tested.
- **From PowerShell** (a quick test):

  ```
  curl.exe http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello! Who are you?\"}]}'
  ```

Thinking: the model thinks before it answers. The launcher uses the model's own default effort (`high`, 0.9). To
change it, set `$env:INKLING_EFFORT` before the start: `none` (no thinking), `minimal`, `low`, `medium`, `high`,
`xhigh`, `max`, or a number from 0.0 to 0.99. An app can also send `reasoning_effort` (one of the words) in each
request. With `none`, the model still writes a short thinking block (about 120 tokens), so give an answer limit of
300 tokens or more.

Sampling: the launcher uses the model maker's settings (temperature 1.0, top-p 1.0, top-k 0, min-p 0). An app can
send its own values in each request.

Saved chats: with Python 3.10 or newer in PATH (no extra packages), the start also runs a small layer in front of
the server. It saves a long chat to disk after each answer, so after a restart the next turn of that chat starts in
seconds, not minutes. Without Python, the server runs plain and the start prints a note; everything else works.

Settings are environment variables, for example `$env:INKLING_CTX = 131072` before the start. The list is at the top
of `start-server.ps1`. By default only this computer can connect. `$env:INKLING_HOST = '0.0.0.0'` opens it to your
network: if you do that, also set `$env:INKLING_KEY_FILE` to a file holding an API key, or anyone on your network can
use your GPU.

## What to try

1. **A long document.** Paste a long text (a book chapter, a long report, a code base: 50k to 200k tokens) and ask
   questions whose answers are in the middle or near the start.
2. **A multi-turn chat with a restart.** Have a long chat (a few turns), stop the server, start it again, and continue
   the same chat. With saved chats on, the next answer should start quickly.
3. **A coding task.** Ask for a function or a small program, run it, and ask for fixes.
4. **A tool or agent task**, if you have a harness that uses tools (function calls).

## What to send back

Copy this into a GitHub issue and fill it in:

```
Card and VRAM:
Driver version (nvidia-smi, top line):
Windows version:
System RAM:
Python for saved chats (yes / no):

From the start output:
  the "kv" line (kv K q4_0 + V q4_0 (8 global caches), tiered: cells ...):
  the "warm" line (VRAM ... MiB):
  any "note" or "SPILL" line:

Speed (logs\server-8080.log has lines like "eval time = ... tokens per second"):
  a short chat (under 4k tokens):        tok/s
  around 32k tokens of context:          tok/s
  around 128k tokens (if you tried):     tok/s
  the longest you tried (how many tokens, how fast):

Errors (the exact text, and the last 30 lines of logs\server-8080.log):

How the answers felt (long-document answers, chat after a restart, code, tools):
```

## Repetition guard

Sometimes the model calls the same tool again and again with nearly the same arguments and gets short results back
(an agent loop). The server then adds one short note to that request only: "You have made nearly the same <tool> call
<N> times in a row. Try a different step or give your answer." The note does not go into the history of your app or
harness. Each time, the server log in `logs\` gets one WARN line that starts with `repetition guard`.

- It is on by default. To turn it off, set `$env:LLAMA_ARG_REPEAT_GUARD = '0'` (or
  `$env:INKLING_EXTRA = '--no-repeat-guard'`) before the start.
- Measured effect: at two stuck points it broke the loop in 6 and 7 of 10 tries, against 0 of 10 without it. It does
  not stop every loop.
- After the guard acts, the next request reads about 4,000 tokens of the prompt again, one time.

## Two slots (opt-in)

By default the server answers one request at a time. To run two requests at the same time (for example two agent
sessions), set `INKLING_SLOTS=2` before the start:

```
$env:INKLING_SLOTS = '2'; powershell -ExecutionPolicy Bypass -File start-server.ps1 -Wait
```

- The two slots share one window of 262,144 tokens (a unified cache): one chat can use all of it, but the two chats
  together cannot use more.
- Fewer K/V cells fit in VRAM (about 21,000 fewer; on an idle RTX 4070 with Windows 228,864 in place of 250,112),
  because each slot keeps its own sliding-window cache. The `kv` line shows the cells and "2 slots".
- Saved conversations (the layer) are off with two slots; the plain server runs.
- A long answer at temperature 0 is not bit-equal to the same request with one slot: where two tokens have almost the
  same probability, the batch layout can change which one comes first. The two chats stay separate (tested: a code
  given to one slot never appears in the answer of the other).
- Two slots do not make one chat faster. When both slots work at the same time, each one is slower (tested on an
  RTX 4070 with Windows: 27 and 38 tok/s with two prompts of about 26,000 and 29,000 tokens; one slot gave about 59
  tok/s on the same prompts). Together the two slots give about 10 % more tokens per second than one slot.
- `INKLING_SLOTS` accepts 1 or 2 only.

## Known limits

- Decode at a very long context is slower: about 30 tok/s at 245k tokens on an RTX 4070 with no display,
  about 58 tok/s at 32k and 42 at 117k (about 65 tok/s for a short chat at 4k). With a desktop on the card less of the context fits in VRAM: at 245k about 22 tok/s with 500 MiB used and 15.5 with
  900 MiB.
- The first read of a very long prompt takes a while: about 3.5 minutes for 245k tokens on an RTX 4070. Later turns of
  the same chat read only the new text.
- Text only. The original model can also read images; the image part is not in this model file.
- No drafting (faster decode with a draft model) in this build: the model's DSpark drafter is not supported yet.
- One chat at a time by default (one slot); two with `INKLING_SLOTS=2` (opt-in, see "Two slots": one shared window, not bit-equal to one slot). The server listens on 127.0.0.1 only.
- On Windows, OpenCode runs shell commands in PowerShell; the model sometimes writes bash syntax there and can get stuck repeating trivial commands. The repetition guard (see above) adds a note then, but it does not always break the loop; if the model still repeats, stop the turn and ask it to use the edit tool.
- Inkling 2 Mini always runs with flash attention. With `-fa off`, the engine turns it on and writes a WARN line in the normal log.
- A display on the same card: the launcher keeps more room for it (600 MiB in place of 250). Memory was
  tested with a simulated desktop load of 500 / 900 MiB (VRAM held by a separate process); not yet with a display on the same card.
- RTX 20 (Turing) cards are not supported. RTX 50 (Blackwell), Hopper, AMD and Apple Silicon: not yet (see "Hardware").
- Use of the model is subject to the Thinking Machines Model Acceptable Use Policy (see NOTICE).

Model updates come as a new .gguf file; the bundle stays the same. Changes: `CHANGELOG.md`.

Licences and credits: `LICENSE`, `NOTICE`, `CREDITS.md`.
