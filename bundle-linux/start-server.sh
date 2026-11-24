#!/usr/bin/env bash
# Kintsugi for Inkling 2 Mini: Thinking Machines' Inkling 2 Mini on one 12 GB NVIDIA card (RTX 30 / 40), Linux
# x86_64: 262,144-token window, tiered K/V cache. The same defaults and checks as start-server.ps1 of the Windows bundle.
#
#   ./start-server.sh [--port 8080] [--wait] [--dry-run]
#
#   --wait     wait until the server answers, send one warm-up request, print the VRAM in use, then return
#   --dry-run  print the settings and the server command line, start nothing
#
# The model file goes in models/Kintsugi-Inkling-2-Mini-12G.gguf (see README.md). It listens on 127.0.0.1 only (this
# computer), unless you set INKLING_HOST. Stop it with: ./stop.sh
#
# Settings (environment variables):
#   INKLING_MODEL         model file (default models/Kintsugi-Inkling-2-Mini-12G.gguf)
#   INKLING_CTX           window in tokens (default 262144; with the tiered cache it is rounded up to a multiple of 32,768)
#   INKLING_KV            K/V cache of the 8 global attention caches: q4_0 (default; K and V q4_0) | q8_0 (K and V q8_0).
#                         The 40 window caches are f16.
#   INKLING_MARGIN        MiB kept free under the card's limit (default 250 with no display on the card, 600 with one)
#   INKLING_CELLS         K/V cells in VRAM (skips the arithmetic below)
#   INKLING_EFFORT        thinking effort: none | minimal | low | medium | high (default; = 0.9, the model's own
#                         default) | xhigh | max, or a number from 0.0 to 0.99 (two decimals at most). "none" = no
#                         thinking.
#   INKLING_HOST          address clients connect to (default 127.0.0.1 = this computer only; 0.0.0.0 = your network:
#                         then set INKLING_KEY_FILE too, or anyone on the network can use your GPU)
#   INKLING_KEY_FILE      API key file; with it, clients must send "Authorization: Bearer <key>"
#   INKLING_LAYER         1 (default) = saved conversations through the layer (needs Python 3.10+); 0 = plain server
#   INKLING_SLOT_SAVE_PATH  folder of the saved conversations (default slots/ in this folder)
#   INKLING_SLOT_CACHE_GB   size limit of that folder in GB (default 20; the oldest files are removed past it)
#   INKLING_UB            micro-batch for prompt reads (default 1024; 512 uses less VRAM and reads slower)
#   INKLING_SLOTS         server slots: 1 (default) | 2 = two requests at the same time (opt-in, README "Two slots"):
#                         one window shared by both (unified cache), fewer K/V cells in VRAM, saved conversations off,
#                         and long answers at temperature 0 are not bit-equal to one slot
#   INKLING_EXTRA         more llama-server arguments
#   INKLING_CHAT_TEMPLATE chat template: "model" (default) = the template inside the model file; any other value = the
#                         path of a template file
#   INKLING_CEILING_MIB   the card's usable limit in MiB (default: read from the driver, see below)
#   INKLING_TEST_DRIVER   replaces the driver version that nvidia-smi gives (tests)
#   INKLING_TEST_CC       replaces the compute capability that nvidia-smi gives (tests; v1.0.1 card check)
set -u
# the bundle folder, resolved (readlink -f): the same path whether the script, the folder or a parent is reached
# through a symlink, so start-server.sh and stop.sh always agree on the layer and engine paths
ROOT="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
PRODUCT="Kintsugi for Inkling 2 Mini"
pick() {   # pick NAME DEFAULT: INKLING_NAME, else DEFAULT
  local a="INKLING_$1"
  if [ -n "${!a:-}" ]; then echo "${!a}"; else echo "$2"; fi
}
PORT=8080; WAIT=0; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --wait) WAIT=1; shift ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown argument: $1 (use --port N, --wait, --dry-run)"; exit 2 ;;
  esac
done

# The model file: Thinking Machines' Inkling 2 Mini weights with the experts in EXL3 trellis code (2 and 3 bits), made
# by us: map M2 (the down projections 3-bit in all layers, gate_up 3-bit in the 4 global layers, 2-bit elsewhere).
# Another file (INKLING_MODEL) uses the memory numbers below, moved by the difference in size.
MODEL_NAME=Kintsugi-Inkling-2-Mini-12G.gguf
MODEL_BYTES=9067638560
MODEL_SHA=7d93d5e8e61f6694776ec8b488751787a7dd3cb745d4f9eaee9f0b78b0e60cec
MODEL_URL=https://huggingface.co/CaryPalmer/Kintsugi-Inkling-2-Mini
MODEL="$(pick MODEL "$ROOT/models/$MODEL_NAME")"
BIN="$ROOT/bin"
CTX="$(pick CTX 262144)"
KV="$(pick KV q4_0 | tr '[:upper:]' '[:lower:]')"
EFFORT="$(pick EFFORT high | tr '[:upper:]' '[:lower:]')"
# The CUDA runtime in bin/ sets the driver floor. This build uses CUDA 12.8, which needs driver 570 or newer.
MIN_DRIVER=570
if [ ! -f "$MODEL" ]; then
  echo "$PRODUCT: model file not found: $MODEL"
  echo "  Download $MODEL_NAME ($MODEL_BYTES bytes) from $MODEL_URL"
  echo "  and put it in $ROOT/models. SHA-256 $MODEL_SHA"
  echo "  (check it with: sha256sum models/$MODEL_NAME). Or set INKLING_MODEL to the file's path."
  exit 1
fi

# The engine of this bundle (bin/llama-server) is used first; a LLAMA_SERVER set in the environment overrides it.
# Do not use a llama-server found on PATH: another build, mixed with bin/ it aborts (GGML_ASSERT in arg.cpp).
if [ -n "${LLAMA_SERVER:-}" ]; then
  echo "using LLAMA_SERVER=$LLAMA_SERVER (not the bundled engine)"
else
  LLAMA_SERVER="$BIN/llama-server"
fi

if [ ! -x "$LLAMA_SERVER" ]; then
  echo "llama-server not found."
  echo "Set LLAMA_SERVER to its full executable path."
  exit 1
fi

command -v nvidia-smi > /dev/null || { echo "nvidia-smi not found: install the NVIDIA driver"; exit 1; }
command -v curl > /dev/null || { echo "curl not found: install curl (the start uses it to check the server)"; exit 1; }
EFFORT_WORDS="none,minimal,low,medium,high,xhigh,max"
# The allow list compares text: a numeric INKLING_EFFORT is added so it passes as a number. The server sends the
# fallback as a JSON string, and the template stops on a string number ("0.5"), so the fallback is always a word: the
# one nearest the effort.
# A number is written in its shortest form ("0.5", not ".5" or "0.50"): the server's allow check compares the JSON text
# of the number. 0 is the word "none".
EFFORT_FALLBACK=$EFFORT; EFFORT_NUM=0
case ",$EFFORT_WORDS," in
  *",$EFFORT,"*) EFFORT_ALLOW=$EFFORT_WORDS ;;
  *) if [[ "$EFFORT" =~ ^(0?\.([0-9]{1,2})|0(\.([0-9]{0,2}))?)$ ]]; then
       FRAC="${BASH_REMATCH[2]}${BASH_REMATCH[4]}"; FRAC="${FRAC%0}"; FRAC="${FRAC%0}"
       if [ -z "$FRAC" ]; then EFFORT=none; EFFORT_FALLBACK=none; EFFORT_ALLOW=$EFFORT_WORDS
       else
       EFFORT="0.$FRAC"; EFFORT_NUM=1
       EFFORT_ALLOW="$EFFORT_WORDS,$EFFORT"
       EFFORT_FALLBACK=$(awk -v e="$EFFORT" 'BEGIN { split("none minimal low medium high xhigh", w, " "); split("0 0.1 0.2 0.7 0.9 0.99", v, " ");
         best = 1; for (i = 2; i <= 6; i++) { d = v[i] - e; if (d < 0) d = -d; b = v[best] - e; if (b < 0) b = -b; if (d < b) best = i }; print w[best] }')
       fi
     else
       echo "INKLING_EFFORT must be one of $EFFORT_WORDS or a number from 0.0 to 0.99 (at most two decimals)"; exit 1
     fi ;;
esac

# ---- NVIDIA driver ------------------------------------------------------------------------------------------------------
DRIVER="$(pick TEST_DRIVER '')"
[ -z "$DRIVER" ] && DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2> /dev/null | head -1 | tr -d ' ')"
if [[ "$DRIVER" =~ ^([0-9]+)\. ]]; then
  if [ "${BASH_REMATCH[1]}" -lt "$MIN_DRIVER" ]; then
    echo "NVIDIA driver $DRIVER is too old: $PRODUCT needs driver $MIN_DRIVER or newer (CUDA 12.8). Update the driver, then start again."
    exit 1
  fi
else
  echo "WARN   could not read the NVIDIA driver version from nvidia-smi: the driver check is skipped"
fi

# A driver the runtime cannot use (or no CUDA device) does not stop the engine: it runs on the CPU, very slowly, and still
# answers. no_gpu_message LOG prints the stop message when the server log shows that.
no_gpu_message() {
  [ -f "$1" ] && grep -q -F -e 'failed to initialize CUDA' -e 'no usable GPU found' "$1" \
    && echo "the engine could not use the GPU (see $1): update the NVIDIA driver to $MIN_DRIVER or newer"
}
SP=""; LP=""
# stop_fail MESSAGE: the start failed. Stop the engine and the layer that this start made, remove their pid files, print
# MESSAGE, exit 1. A layer left behind keeps the client port, and the next start fails with "Address already in use"; an
# engine left behind holds its VRAM. Both are easy to miss when the script is run once and its exit code is not read.
stop_fail() {
  [ -n "$SP" ] && kill -9 "$SP" 2> /dev/null
  [ -n "$LP" ] && kill -9 "$LP" 2> /dev/null
  [ -n "${LOGS:-}" ] && rm -f "$LOGS/server-$PORT.pid" "$LOGS/layer-$PORT.pid"
  echo "$1"
  exit 1
}

# ---- Card generation (v1.0.1): bin/ has native code for compute capability 7.5 / 8.6 / 8.9 (RTX 20 / 30 / 40), 9.0
# (H100 / H200) and 12.0 (RTX 50). CUDA loads only the code of your card, so the code for the other cards costs no VRAM
# or speed. bin/ also has portable code (compute_90 PTX), but this engine does not set its kernel choices for a card that
# runs it (the Kintsugi-27B v1.1.2 fix F1 is not in it), so such cards (for example compute capability 10.0 or 12.1) stop
# here. INKLING_TEST_CC=<x.y> replaces the value that nvidia-smi gives (tests).
CC="$(pick TEST_CC '')"
[ -z "$CC" ] && CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2> /dev/null | head -1 | tr -d ' ')"
SUPPORTED="RTX 30 / 40 / 50 cards with 12 GB or more, and H100 / H200"
if [[ "$CC" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
  CCMAJ=${BASH_REMATCH[1]}; CCMIN=${BASH_REMATCH[2]}; CCN=$((10 * CCMAJ + CCMIN))
  if [ "$CCN" -lt 75 ]; then
    echo "This NVIDIA card (compute capability $CC) is older than RTX 20 (Turing, 7.5): $PRODUCT has no code for it. Supported: $SUPPORTED."
    exit 1
  elif { [ "$CCMAJ" = 7 ] && [ "$CCN" -ne 75 ]; } || { [ "$CCMAJ" = 8 ] && [ "$CCMIN" -lt 6 ]; }; then
    echo "This NVIDIA card (compute capability $CC, for example a V100 / A100 / A30) has no code in this bundle. Supported: $SUPPORTED."
    exit 1
  elif [ "$CCN" -ge 90 ] && [ "$CCN" -ne 90 ] && [ "$CCN" -ne 120 ]; then
    echo "This NVIDIA card (compute capability $CC) has no native code in this bundle, and $PRODUCT does not run the portable code in this version. Supported: $SUPPORTED."
    exit 1
  elif [ "$CCN" -eq 75 ]; then
    echo "note   RTX 20 (Turing, compute capability 7.5) is not tested with $PRODUCT in this version"
  fi
else
  echo "WARN   could not read the card's compute capability from nvidia-smi: the card check is skipped"
fi

# ---- VRAM line: how many K/V cells fit in VRAM; the rest of the window lives in system RAM -----------------------------
# Inkling 2 Mini runs layers 6-17 three times (the loop), so each token has 48 attention applications, each with its own
# K/V: 8 global (layer 5, layer 11 x3, layer 17 x3, layer 23) and 40 sliding-window (window 1,024).
#  - The 8 global caches hold the long context: tiered (VRAM head + pinned system-RAM tail), type INKLING_KV.
#    Per cell and global application: K 8 heads x 128 + V 8 heads x 128 = 2,048 values.
#    q8_0: 8 x 1,024 x 34 / 32 = 8,704 bytes of K (or of V) per cell, 17,408 (17 KiB) K + V; q4_0: 4,608.
#  - The 40 sliding-window caches hold 1,024 + micro-batch cells each, f16, all in VRAM: a fixed cost (below).
# INKLING_KV modes (K type / V type of the global caches):
#   q4_0 (default)   K q4_0 + V q4_0: 4,608 + 4,608 = 9,216 bytes per cell (v1.1; gates: docs/INKLING-DEEP-DECODE.md 10)
#   q8_0             K q8_0 + V q8_0: 17,408 bytes per cell (the v1.0 default)
#   q8q4             K q8_0 + V q4_0: 13,312 bytes per cell = more of the context in VRAM. NOT in this release (a later
#                    version, labelled with its measured KL and top-1); kept off and out of the docs. Needs kernel work; set MIXED_READY=1 when it lands.
MIXED_READY=0
NGLOBAL=8
case "$KV" in
  q4_0) KTYPE=q4_0; VTYPE=q4_0; CELL=$((4608 + 4608)) ;;
  q8_0) KTYPE=q8_0; VTYPE=q8_0; CELL=$((8704 + 8704)) ;;
  q8q4) [ $MIXED_READY = 1 ] || { echo "INKLING_KV=q8q4 (K q8_0 + V q4_0) is not in this build yet; use q4_0 or q8_0"; exit 1; }
        KTYPE=q8_0; VTYPE=q4_0; CELL=$((8704 + 4608)) ;;
  *) echo "INKLING_KV must be q4_0 or q8_0 in this build"; exit 1 ;;
esac
UB="$(pick UB 1024)"
# Server slots (INKLING_SLOTS): 1 (default) or 2. Two slots run as -np 2 --kv-unified: one pool of cells and one window
# that either slot can fill (the slots share it). Gates: docs/INKLING-DEEP-DECODE.md 12.3 / 12.3b (engine np2).
SLOTS="$(pick SLOTS 1)"
case "$SLOTS" in 1|2) ;; *) echo "INKLING_SLOTS must be 1 or 2"; exit 1 ;; esac
# Sliding-window caches: 40 x (1,024 x slots + micro-batch) cells x 2,048 values x 2 bytes (f16) = 320 MiB at -ub 1024,
# 240 at 512 (one slot); 480 MiB at -ub 1024 with two slots (each slot keeps its own last 1,024 tokens).
SWA_MIB=$(( (40 * (1024 * SLOTS + UB) * 4096 + 1048575) / 1048576 ))
# Short-convolution state: one per slot, 48 layers x 6,656 channels x 3 steps x 4 bytes = 3.7 MiB; FIXED holds one slot's,
# so a second slot adds ceil(2 x 3.66) - ceil(3.66) = 4 MiB.
RS_MIB=0
[ "$SLOTS" -gt 1 ] && RS_MIB=$(( (SLOTS * 48 * 6656 * 3 * 4 + 1048575) / 1048576 - (48 * 6656 * 3 * 4 + 1048575) / 1048576 ))
IFS=',' read -r TOTAL USED DISP < <(nvidia-smi --query-gpu=memory.total,memory.used,display_active --format=csv,noheader,nounits | head -1)
TOTAL=$(echo "$TOTAL" | tr -dc '0-9'); USED=$(echo "$USED" | tr -dc '0-9'); DISP=$(echo "$DISP" | tr -d ' ')
# A display on the card = a graphics process (type G or C+G) in nvidia-smi's process table. display_active is not used:
# a headless server GPU can report "Enabled" (seen on a rented RTX 4070, driver 580). INKLING_MARGIN overrides.
GFX=$(nvidia-smi 2> /dev/null | grep -c -E '^\|[[:space:]]+[0-9]+[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+[0-9]+[[:space:]]+(G|C\+G)[[:space:]]')
HEADLESS=1; [ "${GFX:-0}" -gt 0 ] && HEADLESS=0
[ "$TOTAL" -lt 11500 ] && echo "note   this card has $TOTAL MiB; $PRODUCT is made for 12 GB cards and may not load"
MARGIN="$(pick MARGIN $([ $HEADLESS = 1 ] && echo 250 || echo 600))"
if [ -n "$(pick MARGIN '')" ]; then MARGIN_WHY="INKLING_MARGIN"; elif [ $HEADLESS = 1 ]; then MARGIN_WHY="no display on this card"; else MARGIN_WHY="display on this card: a graphics process in nvidia-smi; INKLING_MARGIN=250 if that is wrong"; fi
# Fixed footprint at load without the global K/V cells, the staging buffer and the sliding-window caches, for the
# MODEL_BYTES file; a model file of another size moves it by the difference.
# Measured on this file (RTX 4070, Windows, G4 run k4b, engine 27b5ee563, the launcher with both warm-ups), without the
# sliding-window caches: FIXED_MIB = (VRAM used after load) - (VRAM used before start) - SWA MiB - cells x cell / 1 MiB
# - staging MiB. Linux reads its own (larger) driver reservation below, so it keeps the same room under its limit.
FIXED_UB1024=8916   # FIXED_MIB at -ub 1024
FIXED_UB512=8793    # FIXED_MIB at -ub 512
FIXED_BASE=$FIXED_UB1024; [ "$UB" -le 512 ] && FIXED_BASE=$FIXED_UB512
SIZE=$(stat -L -c %s "$MODEL")   # -L: the size of the model file itself, also when $MODEL is a symlink
HEAD_MIB=$(( (SIZE - MODEL_BYTES) / 1048576 ))
FIXED=$((FIXED_BASE + HEAD_MIB + SWA_MIB + RS_MIB))
# Growth: from after load to the flat level after the first varied requests. The two warm-up requests take 52 MiB; the
# first 50 varied prompts then add up to 250 more once (CUDA graphs and buffers of new lengths), and VRAM stays flat
# after that. Growth = 52 + 250, rounded up = 300 (G4 runs B1 and B3 on M2). The margin is on top of it.
GROWTH=300
# The driver keeps part of the card for itself ("Reserved" in nvidia-smi -q; 416 MiB on an RTX 4070 with Linux driver 595,
# 272 MiB with Windows). No program can use it: past the rest, an allocation fails (out of memory). Ceiling = the smaller
# of 97.7 % of the card and the card less the reserved part less 8 MiB. If nvidia-smi does not show the reserved part,
# 512 MiB is assumed. INKLING_CEILING_MIB overrides.
RESERVED=$(nvidia-smi -q -d MEMORY -i 0 2> /dev/null | awk '/FB Memory Usage/ { fb = 1; next } fb && /Memory Usage/ { exit } fb && $1 == "Reserved" { print $3; exit }' | tr -dc '0-9')
if [ -z "$RESERVED" ]; then
  RESERVED=512
  echo "note   nvidia-smi does not show the card's reserved memory: 512 MiB assumed (INKLING_CEILING_MIB overrides)"
fi
CEIL_A=$((TOTAL * 977 / 1000)); CEIL_B=$((TOTAL - RESERVED - 8)); [ "$CEIL_B" -lt "$CEIL_A" ] && CEIL_A=$CEIL_B
CEILING="$(pick CEILING_MIB $CEIL_A)"
# VRAM of the global K/V = cells x cell + staging; staging = one global application's K and V of the cells in system
# RAM = (window - cells) x cell / 8. So cells = (budget - window x cell / 8) / (cell x 7 / 8), in steps of 256.
# The engine has one VRAM staging buffer for K and one for V, sized to one global application's tail (confirmed for
# engine ace85a829; G4 load predicted within 15 MiB).
# The full formula (all MiB unless said; G4 checks it against the "kv" line):
#   limit   = min(card total, ceiling); ceiling = min(floor(total x 0.977), total - reserved - 8)
#   fixed   = FIXED(ub) + (model file bytes - profile bytes) / 1 MiB + SWA MiB + RS MiB (FIXED excludes the SWA caches;
#             SWA MiB and RS MiB as above: 320 + 0 with one slot, 480 + 4 with two at -ub 1024)
#   budget  = (limit - used before start - margin - fixed - growth) x 1 MiB - window x cell / 8      (bytes)
#   cells   = floor(budget / (cell x 7 / 8) / 256) x 256, at least 3,328, at most the window
# Worked example, idle headless RTX 4070 on Windows, the release defaults (q8_0: cell 17,408 B; ub 1024; window 262,144):
#   limit 11,999 (total 12,282, reserved 272); margin 250; fixed 8,916 + 0 + 320 = 9,236; growth 300
#   budget = (11,999 - 0 - 250 - 9,236 - 300) MiB - 262,144 x 17,408 / 8 B = 2,213 MiB - 544 MiB = 1,669 MiB
#   cells  = 1,750,073,344 / 15,232 = 114,894 -> 114,688 (a multiple of 256)
#   Other cases (same card): 500 MiB already used 80,384; 900 MiB used 52,736; -ub 512 headless 128,768.
#   q4_0 (cell 9,216 B), the same card: one slot 250,112 cells (fixed 9,236); two slots (INKLING_SLOTS=2) fixed 8,916 +
#   0 + 480 + 4 = 9,400, budget (11,999 - 250 - 9,400 - 300) - 288 = 1,761 MiB -> 228,864 cells (gate MEM2).
cells_for() {   # cells_for FIXED: K/V cells that fit in VRAM
  local lim=$TOTAL; [ "$CEILING" -lt "$lim" ] && lim=$CEILING
  local budget=$(( (lim - USED - MARGIN - $1 - GROWTH) * 1048576 - CTX * CELL / NGLOBAL ))
  local c=$(( budget * NGLOBAL / (CELL * (NGLOBAL - 1)) / 256 * 256 ))
  [ "$c" -lt 3328 ] && c=3328
  [ "$c" -gt "$CTX" ] && c=$CTX
  echo "$c"
}
CELLS="$(pick CELLS "$(cells_for $FIXED)")"
[ "$CELLS" -gt "$CTX" ] && CELLS=$CTX
# Window guard: with the tiered cache (cells in VRAM < window) the window is always a multiple of 32,768 tokens. Prompt
# reads past the VRAM line are wrong otherwise (a known bug of the tiered K/V). Any other window is rounded UP (and to
# at most 262,144 when it was not above it); with no tier (all cells in VRAM) any window is fine.
if [ "$CELLS" -lt "$CTX" ] && [ $((CTX % 32768)) -ne 0 ]; then
  OLD=$CTX
  CTX=$(( (CTX + 32767) / 32768 * 32768 ))
  [ "$OLD" -le 262144 ] && [ "$CTX" -gt 262144 ] && CTX=262144
  echo "note   window $OLD rounded up to $CTX (a multiple of 32,768 with the tiered K/V cache)"
  CELLS="$(pick CELLS "$(cells_for $FIXED)")"; [ "$CELLS" -gt "$CTX" ] && CELLS=$CTX
fi
[ "$CELLS" -le 3328 ] && echo "note   only $CELLS K/V cells fit in VRAM: decode reads most of the K/V cache from system RAM (close other programs that use the card)"

# ---- Engine switches -------------------------------------------------------------------------------------------------
# No drafting in this release: the DSpark drafter of Inkling 2 Mini is not supported by this engine yet.
export GGML_CUDA_BATCH_INVARIANT=1
export LLAMA_ARG_CHECKPOINT_EVERY_NT=8192
# The 40 sliding-window caches stay f16 whatever -ctk / -ctv say (they are small and fixed).
# It is an environment variable only (no CLI flag in engine ace85a829).
export LLAMA_INKLING_SWA_KV="${LLAMA_INKLING_SWA_KV:-f16}"
export GGML_CUDA_POOL_RESERVE_MIB="${GGML_CUDA_POOL_RESERVE_MIB:-96}"
# Read path of the trellis experts for long prompts: the exact read only in this release (the int8 fast read is not
# tested on Inkling).
export GGML_ESCHA_PREFILL=recon
export GGML_CUDA_DEDICATED_CEILING_MIB="${GGML_CUDA_DEDICATED_CEILING_MIB:-$CEILING}"
if [ $EFFORT_NUM = 1 ]; then export LLAMA_ARG_CHAT_TEMPLATE_KWARGS="{\"reasoning_effort\":$EFFORT}"
else export LLAMA_ARG_CHAT_TEMPLATE_KWARGS="{\"reasoning_effort\":\"$EFFORT\"}"; fi
export LD_LIBRARY_PATH="$BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"   # the engine and the CUDA runtime libraries are in bin/

KEY_FILE="$(pick KEY_FILE '')"
LISTEN="$(pick HOST 127.0.0.1)"
if [ "$LISTEN" != 127.0.0.1 ] && [ "$LISTEN" != localhost ] && [ -z "$KEY_FILE" ]; then
  echo "WARNING listening on $LISTEN with NO API key: anyone who can reach this computer can use the server (and your GPU). Set INKLING_KEY_FILE to a file with a key."
fi

# ---- Saved conversations: the layer (layer/bonsai_layer.py) runs in front of the server on the client port; the
# ---- server moves to 127.0.0.1:<port + 10000>. It needs Python 3.10 or newer (standard library only).
USE_LAYER=0; [ "$(pick LAYER 1)" = 1 ] && USE_LAYER=1
# The layer serves one server slot (it saves and restores slot 0 and runs one request at a time): two slots run the
# plain server.
if [ $USE_LAYER = 1 ] && [ "$SLOTS" -gt 1 ]; then
  echo "note   $SLOTS slots: saved conversations are off (the layer serves one server slot); the plain server runs"
  USE_LAYER=0
fi
PY=""
if [ $USE_LAYER = 1 ]; then
  PY="$(command -v python3 || command -v python || true)"
  if [ -z "$PY" ]; then
    echo "note   no python3 in PATH: saved conversations are off (the plain server runs). Install Python 3.10+ to use them, or set INKLING_LAYER=0 to hide this note."
    USE_LAYER=0
  elif ! "$PY" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2> /dev/null; then
    echo "note   saved conversations are off: they need Python 3.10 or newer (the python in PATH is older or does not run). The plain server runs."
    USE_LAYER=0
  fi
fi
SERVER_PORT=$PORT
SLOT_DIR=""
if [ $USE_LAYER = 1 ]; then
  SERVER_PORT=$((PORT + 10000))
  SLOT_DIR="$(pick SLOT_SAVE_PATH "$ROOT/slots")"
  [ $DRY = 0 ] && mkdir -p "$SLOT_DIR" && SLOT_DIR="$(cd "$SLOT_DIR" && pwd -P)"
fi
TEMPLATE="$(pick CHAT_TEMPLATE model)"
if [ "$TEMPLATE" != model ] && [ ! -f "$TEMPLATE" ]; then echo "chat template not found: $TEMPLATE (set INKLING_CHAT_TEMPLATE)"; exit 1; fi
SRV_HOST=$LISTEN; [ $USE_LAYER = 1 ] && SRV_HOST=127.0.0.1
# Sampling: the model vendor's defaults (temperature 1.0, top-p 1.0, top-k 0, min-p 0); a request can override them.
# Reasoning budget: the Inkling parser marks the thinking block (<|content_thinking|> .. <|end_message|>); at the budget
# the server closes it with the message below and the model writes its answer.
# Checked on the release build (G5 part a): the tags are single control tokens (200008, 200010); a 64-token budget
#   closed the thinking at 70 tokens and the answer came back.
# --backend-sampling has no effect while --reasoning-budget is set: the engine turns it off on every chat request
# ("backend sampling is not compatible with reasoning budget"). Kept: the gates (G3, G4) ran with it.
# --kq-mask-packed has no effect on Inkling (its graph uses dense masks; the banded op uses none); kept, harmless.
# --kv-vram-cells applies only to the global (base) cache; the sliding-window caches stay all in VRAM.
# -np: the slots; two slots add --kv-unified (one pool of cells, one shared window).
ARGS=(-m "$MODEL" -ngl 99 -c "$CTX" -fa on -np "$SLOTS" -ub "$UB" --fit off --jinja
      -ctk "$KTYPE" -ctv "$VTYPE" --kv-vram-cells "$CELLS" --kq-mask-packed --cache-ram 4096 --backend-sampling
      --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0
      --reasoning-budget 20480 -n 24576 --reasoning-budget-message "Now produce the complete answer."
      --reasoning-effort-allow "$EFFORT_ALLOW" --reasoning-effort-fallback "$EFFORT_FALLBACK" --reasoning-max-tokens-floor 24576
      --host "$SRV_HOST" --port "$SERVER_PORT")
[ "$SLOTS" -gt 1 ] && ARGS+=(--kv-unified)
[ "$TEMPLATE" != model ] && ARGS+=(--chat-template-file "$TEMPLATE")
[ -n "$KEY_FILE" ] && ARGS+=(--api-key-file "$KEY_FILE")
[ -n "$SLOT_DIR" ] && ARGS+=(--slot-save-path "$SLOT_DIR")
EXTRA="$(pick EXTRA '')"
[ -n "$EXTRA" ] && read -r -a EXTRA_ARGS <<< "$EXTRA" && ARGS+=("${EXTRA_ARGS[@]}")

echo "$PRODUCT"
echo "model  $(basename "$MODEL")"
echo "listen $LISTEN:$PORT$([ $USE_LAYER = 1 ] && echo " (layer; llama-server on 127.0.0.1:$SERVER_PORT)")$([ -n "$KEY_FILE" ] && echo ", API key on")"
echo "kv     K $KTYPE + V $VTYPE (8 global caches), tiered: cells 0..$CELLS in VRAM, $CELLS..$CTX in system RAM; 40 window caches f16 $SWA_MIB MiB (free $((TOTAL - USED)) MiB, reserved $RESERVED, ceiling $CEILING, fixed $FIXED, growth $GROWTH, margin $MARGIN ($MARGIN_WHY))$([ "$SLOTS" -gt 1 ] && echo "; $SLOTS slots (unified cache)")"
[ "$SLOTS" -gt 1 ] && echo "note   $SLOTS slots share the $CTX-token window (unified cache); long answers at temperature 0 are not bit-equal to one slot (near ties flip with the batch layout)"
echo "template $([ "$TEMPLATE" = model ] && echo "the model file's own" || echo "$TEMPLATE")"
echo "think  effort $EFFORT; sampling temperature 1.0, top-p 1.0, top-k 0, min-p 0; no drafting"
if [ $DRY = 1 ]; then echo "kwargs $LLAMA_ARG_CHAT_TEMPLATE_KWARGS"; echo "args   ${ARGS[*]}"; exit 0; fi

LOGS="$ROOT/logs"; mkdir -p "$LOGS"
LOG="$LOGS/server-$PORT.log"
setsid nohup "$LLAMA_SERVER" "${ARGS[@]}" > "$LOG.out" 2> "$LOG" < /dev/null &
SP=$!
echo "$SP" > "$LOGS/server-$PORT.pid"
echo "pid    $SP   log $LOG"
LAYER_LOG="$LOGS/layer-$PORT.log"
if [ $USE_LAYER = 1 ]; then
  LAYER_KEY=""; [ -n "$KEY_FILE" ] && LAYER_KEY="$(tr -d '\r\n' < "$KEY_FILE")"
  # The stray-tag filter stays off: its rules are for Qwen-style tags, and the Inkling parser drops malformed tails.
  BONSAI_SLOT_SAVE_PATH="$SLOT_DIR" BONSAI_SLOT_CACHE_GB="$(pick SLOT_CACHE_GB 20)" BONSAI_SLOT_CACHE_AFTER=1 BONSAI_LAYER_KEY="$LAYER_KEY" \
    setsid nohup "$PY" "$ROOT/layer/bonsai_layer.py" --host "$LISTEN" --port "$PORT" --upstream "http://127.0.0.1:$SERVER_PORT" \
      --no-interpreter --no-cards --no-lint --no-example-check --no-finish-note --no-input-file --no-strip-stray-tags \
      > "$LAYER_LOG.out" 2> "$LAYER_LOG" < /dev/null &
  LP=$!
  echo "$LP" > "$LOGS/layer-$PORT.pid"
  echo "layer  pid $LP   log $LAYER_LOG   clients use :$PORT; saved conversations in $SLOT_DIR"
fi
echo "stop   $ROOT/stop.sh"
alive() { kill -0 "$1" 2> /dev/null; }
if [ $WAIT = 0 ]; then
  # No wait for the load: the engine writes its GPU set-up to the log before the model load starts.
  for i in $(seq 1 60); do
    { ! alive "$SP" || grep -q -F 'load_model: loading model' "$LOG" 2> /dev/null; } && break
    sleep 1
  done
  M="$(no_gpu_message "$LOG")"; [ -n "$M" ] && stop_fail "$M"
  alive "$SP" || stop_fail "server stopped during start: see $LOG"
  exit 0
fi
AUTH=(); [ -n "$KEY_FILE" ] && AUTH=(-H "Authorization: Bearer $(tr -d '\r\n' < "$KEY_FILE")")
for i in $(seq 1 150); do
  if curl -sf -m 3 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    M="$(no_gpu_message "$LOG")"; [ -n "$M" ] && stop_fail "$M"
    # Warm-up, two requests in at most 600 s: (1) a chat request (about 1,100 tokens in, 8 out, no thinking); (2) about
    # 100 decoded tokens on the server port (/completion, about 200 tokens in): longer decodes take more first-use memory
    # (CUDA graphs, buffers; G4 saw +250 MiB in the first decodes after a chat-only warm-up). The engine takes that
    # memory now, so the VRAM printed below is the peak, and a shortage shows here and not later.
    TEXT="$(for n in $(seq 1 110); do printf 'Entry %d of the log has nothing to report. ' "$n"; done)"
    BODY="{\"model\":\"warmup\",\"max_tokens\":8,\"temperature\":0,\"cache_prompt\":false,\"chat_template_kwargs\":{\"reasoning_effort\":\"none\"},\"messages\":[{\"role\":\"user\",\"content\":\"${TEXT}Say OK.\"}]}"
    T0=$(date +%s.%N)
    curl -sf -m 600 ${AUTH[@]+"${AUTH[@]}"} -H 'Content-Type: application/json' -d "$BODY" "http://127.0.0.1:$PORT/v1/chat/completions" > /dev/null \
      || echo "note   the warm-up request failed"
    TEXT2="$(for n in $(seq 1 20); do printf 'Entry %d of the log has nothing to report. ' "$n"; done)"
    BODY2="{\"prompt\":\"${TEXT2}Entry 21 of the log\",\"n_predict\":100,\"temperature\":0,\"cache_prompt\":false,\"ignore_eos\":true}"
    LEFT=$(awk -v a="$T0" -v b="$(date +%s.%N)" 'BEGIN { l = int(600 - (b - a)); if (l < 30) l = 30; print l }')
    curl -sf -m "$LEFT" ${AUTH[@]+"${AUTH[@]}"} -H 'Content-Type: application/json' -d "$BODY2" "http://127.0.0.1:$SERVER_PORT/completion" > /dev/null \
      || echo "note   the second warm-up request (decode) failed"
    T1=$(date +%s.%N)
    VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
    PROC=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2> /dev/null | awk -F', *' -v p="$SP" '$1 == p { print $2 }')
    echo "warm   $(awk -v a="$T0" -v b="$T1" 'BEGIN { printf "%.1f", b - a }') s; VRAM $VRAM MiB (ceiling $CEILING), server process ${PROC:-?} MiB"
    # Linux has no silent move of GPU memory to system RAM (an allocation that does not fit fails); the engine's own
    # spill lines are still checked.
    if grep -q 'memory_guard: spill' "$LOG" 2> /dev/null; then
      echo "SPILL  the engine reports GPU memory in system RAM: start again with a larger INKLING_MARGIN (now $MARGIN) or fewer INKLING_CELLS (now $CELLS)."
    fi
    echo "ready  http://127.0.0.1:$PORT/v1"
    exit 0
  fi
  if ! alive "$SP"; then
    M="$(no_gpu_message "$LOG")"; [ -n "$M" ] && stop_fail "$M"
    stop_fail "server stopped during load: see $LOG"
  fi
  if [ -n "$LP" ] && ! alive "$LP"; then stop_fail "layer stopped: see $LAYER_LOG"; fi
  sleep 2
done
stop_fail "not ready after 300 s: see $LOG"
