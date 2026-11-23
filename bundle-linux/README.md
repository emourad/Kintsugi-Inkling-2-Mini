# Kintsugi for Inkling 2 Mini v1.0 for Linux (x86_64, RTX 30 / 40, 12 GB)

![Kintsugi for Inkling 2 Mini](inkling-card.png)

Kintsugi for Inkling 2 Mini is Thinking Machines' Inkling 2 Mini (a 26.4B looped mixture-of-experts model) in our own
2-3 bit model file, built to run on one 12 GB NVIDIA card with a 262,144-token context. The engine keeps the
long-context cache at 4 bits (q4_0): on a free card up to 250k tokens stay in VRAM, and with a desktop on the card the
rest goes to system RAM. This is the Linux bundle: the same model file, engine
features and defaults as the Windows bundle. Problems or numbers from your card: please open an issue at
https://github.com/professorpalmer/Kintsugi-Inkling-2-Mini/issues.

## What you need

- An NVIDIA RTX 30 or 40 series card with 12 GB of VRAM or more. RTX 20 (Turing) is not
  supported; RTX 50 (Blackwell) not yet (see "Hardware").
- NVIDIA driver 570 or newer. The engine uses the CUDA 12.8 runtime (in `bin/`). On drivers older than the 580
  family, the engine uses another type of pinned system memory for the long context, and writes one WARN line about it.
- Linux x86_64 with glibc 2.34 or newer and the libstdc++ of GCC 12 or newer (the engine needs GLIBC_2.34 and
  GLIBCXX_3.4.30). Ubuntu 22.04 and 24.04 have both.
- `curl` and `nvidia-smi` (it comes with the driver). 32 GB of system RAM or more is best for very long chats.
- No other libraries are necessary: the bundle carries the CUDA runtime, cuBLAS and the OpenMP runtime (libgomp) in
  `bin/`.
- Optional: Python 3.10 or newer, for saved chats. No extra Python packages are necessary.

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

1. Download `Kintsugi-Inkling-2-Mini-v1.0-linux-x86_64.tar.gz` from the Releases page and unpack it:

   ```
   tar -xzf Kintsugi-Inkling-2-Mini-v1.0-linux-x86_64.tar.gz
   cd Kintsugi-Inkling-2-Mini
   ```

2. Download `Kintsugi-Inkling-2-Mini-12G.gguf` (9.07 GB) from
   https://huggingface.co/CaryPalmer/Kintsugi-Inkling-2-Mini into `models/`, and check it:

   ```
   sha256sum models/Kintsugi-Inkling-2-Mini-12G.gguf
   # 7d93d5e8e61f6694776ec8b488751787a7dd3cb745d4f9eaee9f0b78b0e60cec
   ```

3. Start the server and wait for the line `ready  http://127.0.0.1:8080/v1` (the first start takes about a minute):

   ```
   ./start-server.sh --wait
   ```

4. Chat (below). The server listens only on this computer.
5. Stop it: `./stop.sh`.

## How to chat

- **In the browser:** open http://127.0.0.1:8080/.
  The web page needs a browser (it is served gzip-compressed); for scripts use the API at /v1.
- **From an app:** point any OpenAI-compatible app at `http://127.0.0.1:8080/v1` (model name: anything; no key).
- **Coding harnesses** (Claude Code, Codex, Pi, OpenCode): see `HARNESSES.md` for the settings we tested. Pi needs
  Node.js 20 or newer; Ubuntu 24.04's own Node 18 fails at Pi's start with a SyntaxError.
- **A quick test:**

  ```
  curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hello! Who are you?"}]}'
  ```

Thinking: the launcher uses the model's own default effort (`high`, 0.9). Set `INKLING_EFFORT` to `none` (no
thinking; the model still writes a short thinking block of about 120 tokens, so give an answer limit of 300 tokens or
more), `minimal`, `low`, `medium`, `high`, `xhigh`, `max` or a number from 0.0 to 0.99 to change it. Sampling: the
model maker's settings (temperature 1.0, top-p 1.0, top-k 0, min-p 0); an app can send its own.

Saved chats: with Python 3.10 or newer, the start also runs a small layer in front of the server. It saves a long chat
to disk after each answer, so after a restart the next turn of that chat starts in seconds, not minutes.

Settings are environment variables, for example `INKLING_CTX=131072 ./start-server.sh --wait`. The list is at the
top of `start-server.sh`. By default only this computer can connect; `INKLING_HOST=0.0.0.0` opens it to your network:
then also set `INKLING_KEY_FILE` to a file holding an API key. Logs are in `logs/`.

## Repetition guard

Sometimes the model calls the same tool again and again with nearly the same arguments and gets short results back
(an agent loop). The server then adds one short note to that request only: "You have made nearly the same <tool> call
<N> times in a row. Try a different step or give your answer." The note does not go into the history of your app or
harness. Each time, the server log in `logs/` gets one WARN line that starts with `repetition guard`.

- It is on by default. To turn it off, start with `LLAMA_ARG_REPEAT_GUARD=0 ./start-server.sh --wait` (or set
  `INKLING_EXTRA=--no-repeat-guard`).
- Measured effect: at two stuck points it broke the loop in 6 and 7 of 10 tries, against 0 of 10 without it. It does
  not stop every loop.
- After the guard acts, the next request reads about 4,000 tokens of the prompt again, one time.

## Two slots (opt-in)

By default the server answers one request at a time. To run two requests at the same time (for example two agent
sessions), set `INKLING_SLOTS=2` before the start:

```
INKLING_SLOTS=2 ./start-server.sh --wait
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
- `INKLING_SLOTS` accepts 1 or 2 only. The two-slot numbers above are from Windows.

## Known limits

- Decode at a very long context is slower: about 30 tok/s at 245k tokens on an RTX 4070 with no display (measured on Windows),
  about 58 tok/s at 32k and 42 at 117k (about 65 tok/s for a short chat at 4k). With a desktop on the card less of the context fits in VRAM: at 245k about 22 tok/s with 500 MiB used and 15.5 with
  900 MiB.
- The first read of a very long prompt takes about 3.5 minutes for 245k tokens on an RTX 4070 (measured on Windows); later turns read only the
  new text.
- Text only (the image part of the original model is not in the model file). No drafting in this build.
- One chat at a time by default (one slot); two with `INKLING_SLOTS=2` (opt-in, see "Two slots": one shared window, not bit-equal to one slot).
- On Windows, OpenCode runs shell commands in PowerShell; the model sometimes writes bash syntax there and can get stuck repeating trivial commands. The repetition guard (see above) adds a note then, but it does not always break the loop; if the model still repeats, stop the turn and ask it to use the edit tool.
- Inkling 2 Mini always runs with flash attention. With `-fa off`, the engine turns it on and writes a WARN line in the normal log.
- A display on the same card: the launcher keeps more room for it (600 MiB in place of 250). Memory was
  tested with a simulated desktop load of 500 / 900 MiB (VRAM held by a separate process); not yet with a display on the same card.
- Tested with an RTX 4070 on Ubuntu 24.04, driver 610.57.04, with only the NVIDIA driver (no CUDA toolkit, no NCCL).
  There the launcher keeps 236,288 cells in VRAM (the Linux driver reserves more memory than the Windows one).
- Use of the model is subject to the Thinking Machines Model Acceptable Use Policy (see NOTICE).

Changes: `CHANGELOG.md`. Licences and credits: `LICENSE`, `NOTICE`, `CREDITS.md`.
