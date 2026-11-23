# Kintsugi for Inkling 2 Mini: Thinking Machines' Inkling 2 Mini on one 12 GB NVIDIA card (RTX 30 / 40),
# 262,144-token window, tiered K/V cache.
#
#   powershell -ExecutionPolicy Bypass -File start-server.ps1 [-Port 8080] [-Wait] [-DryRun]
#
#   -Wait     wait until the server answers, send one warm-up request, print the VRAM in use, then return
#   -DryRun   print the settings and the server command line, start nothing
#
# The model file goes in models\Kintsugi-Inkling-2-Mini-12G.gguf (see README.md). It listens on 127.0.0.1 only (this
# computer), unless you set INKLING_HOST. Stop it with: powershell -ExecutionPolicy Bypass -File stop.ps1
#
# Settings (environment variables):
#   INKLING_MODEL         model file (default models\Kintsugi-Inkling-2-Mini-12G.gguf)
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
#   INKLING_LAYER         1 (default) = saved conversations through the layer (needs Python); 0 = plain server
#   INKLING_SLOT_SAVE_PATH  folder of the saved conversations (default slots\ in this folder)
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
param([int]$Port = 8080, [switch]$Wait, [switch]$DryRun)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Product = 'Kintsugi for Inkling 2 Mini'
function Pick($name, $default) {
    $v = [Environment]::GetEnvironmentVariable("INKLING_$name"); if ($v) { return $v }
    $default
}
# A driver too old for CUDA 13 (or no CUDA device) does not stop the engine: it runs on the CPU, very slowly, and still
# answers. Returns the stop message when the server log shows that, else $null.
function Get-NoGpuMessage($LogPath) {
    if (-not (Test-Path $LogPath)) { return $null }
    if (Select-String -Path $LogPath -SimpleMatch -Pattern 'failed to initialize CUDA', 'no usable GPU found' -Quiet) {
        return "the engine could not use the GPU (see $LogPath): update the NVIDIA driver to 580 or newer"
    }
    $null
}
function Stop-NoGpu($Message) {
    if ($p) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    if ($lp) { Stop-Process -Id $lp.Id -Force -ErrorAction SilentlyContinue }
    Write-Host $Message
    exit 1
}

# The size of a file, also when the path is a symbolic link to it (Get-Item gives the link's own length then).
function FileLength($path) {
    $it = Get-Item -LiteralPath $path
    for ($n = 0; $n -lt 8 -and $it.LinkType -eq 'SymbolicLink'; $n++) {
        $t = @($it.Target)[0]
        if (-not [IO.Path]::IsPathRooted($t)) { $t = Join-Path (Split-Path -Parent $it.FullName) $t }
        $it = Get-Item -LiteralPath $t
    }
    $it.Length
}
# ---- The model file: Thinking Machines' Inkling 2 Mini weights with the experts in EXL3 trellis code (2 and 3 bits),
# ---- made by us. Its memory numbers: Fixed = the footprint at load without the K/V cells, the staging buffer and the
# ---- sliding-window caches (at -ub 1024 and 512); Growth = what the first request adds (see below).
# Map M2 (the down projections 3-bit in all layers, gate_up 3-bit in the 4 global layers, 2-bit elsewhere).
# Fixed1024 / Fixed512 / Growth: measured on this file (RTX 4070, Windows, G4 run k4b, engine 27b5ee563, this launcher
#   with both warm-ups); Fixed without the sliding-window caches (formulas below).
$P = @{ Name = 'Kintsugi-Inkling-2-Mini-12G.gguf'; Bytes = 9067638560; Sha = '7d93d5e8e61f6694776ec8b488751787a7dd3cb745d4f9eaee9f0b78b0e60cec'
        Url = 'https://huggingface.co/CaryPalmer/Kintsugi-Inkling-2-Mini'
        Fixed1024 = 8916; Fixed512 = 8793; Growth = 300 }
# Another file (INKLING_MODEL) uses these numbers, moved by the difference in size.
$Model = Pick 'MODEL' (Join-Path $Root "models\$($P.Name)")
if (-not (Test-Path $Model)) {
    Write-Host "${Product}: model file not found: $Model"
    Write-Host "  Download $($P.Name) ($('{0:N0}' -f $P.Bytes) bytes) from $($P.Url)"
    Write-Host "  and put it in $(Join-Path $Root 'models'). SHA-256 $($P.Sha)"
    Write-Host "  (check it with: Get-FileHash models\$($P.Name)). Or set INKLING_MODEL to the file's path."
    exit 1
}
$Bin = Join-Path $Root 'bin'
$Ctx = [int](Pick 'CTX' 262144)
$Kv = (Pick 'KV' 'q4_0').ToLower()
$Effort = (Pick 'EFFORT' 'high').ToLower()
if (-not (Test-Path (Join-Path $Bin 'llama-server.exe'))) { throw "llama-server.exe not found in $Bin" }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw "nvidia-smi not found: install the NVIDIA driver" }
$EffortWords = @('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')
$EffortNum = 0.0
$EffortIsNum = -not ($EffortWords -contains $Effort)
if ($EffortIsNum -and -not ($Effort -match '^(0?\.\d{1,2}|0(\.\d{0,2})?)$' -and [double]::TryParse($Effort, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$EffortNum))) {
    throw "INKLING_EFFORT must be one of $($EffortWords -join ', ') or a number from 0.0 to 0.99 (at most two decimals)"
}
# A number is written in its shortest form ("0.5", not ".5" or "0.50"): the server's allow check compares the JSON text
# of the number. 0 is the word "none".
if ($EffortIsNum) {
    if ($EffortNum -eq 0) { $Effort = 'none'; $EffortIsNum = $false }
    else { $Effort = $EffortNum.ToString('R', [Globalization.CultureInfo]::InvariantCulture) }
}

# ---- NVIDIA driver: the CUDA 13 runtime in bin\ needs driver 580 or newer ---------------------------------------------
$Driver = Pick 'TEST_DRIVER' ''
if (-not $Driver) {
    try { $Driver = "$((& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null) | Select-Object -First 1)".Trim() } catch { $Driver = '' }
}
if ($Driver -match '^(\d+)\.') {
    if ([int]$Matches[1] -lt 580) {
        Write-Host "NVIDIA driver $Driver is too old: $Product needs driver 580 or newer (CUDA 13). Update the driver, then start again."
        exit 1
    }
} else {
    Write-Host "WARN   could not read the NVIDIA driver version from nvidia-smi: the driver check is skipped"
}

# ---- Card generation (v1.0.1): bin\ has native code for compute capability 7.5 / 8.6 / 8.9 (RTX 20 / 30 / 40), 9.0
# (H100 / H200) and 12.0 (RTX 50). CUDA loads only the code of your card, so the code for the other cards costs no VRAM
# or speed. bin\ also has portable code (compute_90 PTX), but this engine does not set its kernel choices for a card that
# runs it (the Kintsugi-27B v1.1.2 fix F1 is not in it), so such cards (for example compute capability 10.0 or 12.1) stop
# here. INKLING_TEST_CC=<x.y> replaces the value that nvidia-smi gives (tests).
$Cc = Pick 'TEST_CC' ''
if (-not $Cc) {
    try { $Cc = "$((& nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2>$null) | Select-Object -First 1)".Trim() } catch { $Cc = '' }
}
if ($Cc -match '^(\d+)\.(\d+)$') {
    $CcMajor = [int]$Matches[1]; $CcMinor = [int]$Matches[2]; $CcN = 10 * $CcMajor + $CcMinor
    $Supported = 'RTX 30 / 40 / 50 cards with 12 GB or more, and H100 / H200'
    if ($CcN -lt 75) {
        Write-Host "This NVIDIA card (compute capability $Cc) is older than RTX 20 (Turing, 7.5): $Product has no code for it. Supported: $Supported."
        exit 1
    } elseif (($CcMajor -eq 7 -and $CcN -ne 75) -or ($CcMajor -eq 8 -and $CcMinor -lt 6)) {
        Write-Host "This NVIDIA card (compute capability $Cc, for example a V100 / A100 / A30) has no code in this bundle. Supported: $Supported."
        exit 1
    } elseif ($CcN -ge 90 -and $CcN -ne 90 -and $CcN -ne 120) {
        Write-Host "This NVIDIA card (compute capability $Cc) has no native code in this bundle, and $Product does not run the portable code in this version. Supported: $Supported."
        exit 1
    } elseif ($CcN -eq 75) {
        Write-Host "note   RTX 20 (Turing, compute capability 7.5) is not tested with $Product in this version"
    }
} else {
    Write-Host "WARN   could not read the card's compute capability from nvidia-smi: the card check is skipped"
}

# ---- VRAM line: how many K/V cells fit in VRAM; the rest of the window lives in system RAM -----------------------------
# Inkling 2 Mini runs layers 6-17 three times (the loop), so each token has 48 attention applications, each with its own
# K/V: 8 global (layer 5, layer 11 x3, layer 17 x3, layer 23) and 40 sliding-window (window 1,024).
#  - The 8 global caches hold the long context: tiered (VRAM head + pinned system-RAM tail), type INKLING_KV.
#    Per cell and global application: K 8 heads x 128 + V 8 heads x 128 = 2,048 values.
#    q8_0 = 34 bytes per 32 values: 8 x 1,024 x 34 / 32 = 8,704 bytes of K (or of V) per cell, 17,408 (17 KiB) K + V.
#    q4_0 = 18 bytes per 32 values: 4,608 bytes of K (or of V) per cell.
#  - The 40 sliding-window caches hold 1,024 + micro-batch cells each, f16, all in VRAM: a fixed cost (below).
# INKLING_KV modes (K type / V type of the global caches):
#   q4_0 (default)   K q4_0 + V q4_0: 4,608 + 4,608 = 9,216 bytes per cell (v1.1; gates: docs/INKLING-DEEP-DECODE.md 10)
#   q8_0             K q8_0 + V q8_0: 17,408 bytes per cell (the v1.0 default)
#   q8q4             K q8_0 + V q4_0: 13,312 bytes per cell = more of the context in VRAM. NOT in this release (a later
#                    version, labelled with its measured KL and top-1); kept off and out of the docs. Needs kernel work; switch $MixedReady on when it lands.
$MixedReady = $false
$NGlobal = 8
$KvModes = @{ 'q4_0' = @('q4_0', 'q4_0', 4608, 4608); 'q8_0' = @('q8_0', 'q8_0', 8704, 8704); 'q8q4' = @('q8_0', 'q4_0', 8704, 4608) }   # K, V, K bytes, V bytes
if (-not $KvModes.ContainsKey($Kv)) { throw "INKLING_KV must be q4_0 or q8_0 in this build" }
if ($Kv -eq 'q8q4' -and -not $MixedReady) { throw "INKLING_KV=q8q4 (K q8_0 + V q4_0) is not in this build yet; use q4_0 or q8_0" }
$KType = $KvModes[$Kv][0]; $VType = $KvModes[$Kv][1]
$Cell = $KvModes[$Kv][2] + $KvModes[$Kv][3]
$Ub = [int](Pick 'UB' 1024)
# Server slots (INKLING_SLOTS): 1 (default) or 2. Two slots run as -np 2 --kv-unified: one pool of cells and one window
# that either slot can fill (the slots share it). Gates: docs/INKLING-DEEP-DECODE.md 12.3 / 12.3b (engine np2).
$SlotsText = Pick 'SLOTS' '1'
if ($SlotsText -notmatch '^[12]$') { throw "INKLING_SLOTS must be 1 or 2" }
$Slots = [int]$SlotsText
# Sliding-window caches: 40 x (1,024 x slots + micro-batch) cells x 2,048 values x 2 bytes (f16) = 320 MiB at -ub 1024,
# 240 at 512 (one slot); 480 MiB at -ub 1024 with two slots (each slot keeps its own last 1,024 tokens).
$SwaMiB = [int][math]::Ceiling(40 * (1024 * $Slots + $Ub) * 2048 * 2 / 1MB)
# Short-convolution state: one per slot, 48 layers x 6,656 channels x 3 steps x 4 bytes = 3.7 MiB; FIXED holds one slot's,
# so a second slot adds ceil(2 x 3.66) - ceil(3.66) = 4 MiB.
$RsMiB = if ($Slots -gt 1) { [int]([math]::Ceiling($Slots * 48 * 6656 * 3 * 4 / 1MB) - [math]::Ceiling(48 * 6656 * 3 * 4 / 1MB)) } else { 0 }
$q = (& nvidia-smi --query-gpu=memory.total,memory.used,display_active --format=csv,noheader,nounits | Select-Object -First 1).Split(',')
$TotalMiB = [int]$q[0]; $UsedMiB = [int]$q[1]; $Headless = $q[2].Trim() -ne 'Enabled'
if ($TotalMiB -lt 11500) { Write-Host "note   this card has $TotalMiB MiB; $Product is made for 12 GB cards and may not load" }
# Margin under the card's limit, on top of the measured growth (a display on the card moves by itself, so more margin).
$Margin = [int](Pick 'MARGIN' $(if ($Headless) { 250 } else { 600 }))
$MarginWhy = if (Pick 'MARGIN' '') { 'INKLING_MARGIN' } elseif ($Headless) { 'no display on this card' } else { 'display on this card: nvidia-smi display_active; INKLING_MARGIN=250 if that is wrong' }
# Fixed footprint at load (the model file's profile above); a file of another size moves it by the difference.
# How the gate measures it (fresh server, no request yet, INKLING_LAYER=0), at -ub 1024 and at -ub 512:
#   FIXED_MIB = (VRAM used after load) - (VRAM used before start) - SWA MiB - cells x cell / 1 MiB - staging MiB
#   with staging MiB = (window - cells) x cell / 8 / 1 MiB (see the formula of CellsFor).
$FixedBase = if ($Ub -le 512) { $P.Fixed512 } else { $P.Fixed1024 }
$HeadMiB = [int][math]::Round(((FileLength $Model) - $P.Bytes) / 1MB)
$Fixed = $FixedBase + $HeadMiB + $SwaMiB + $RsMiB
# Growth: from after load to the flat level after the first varied requests. The two warm-up requests take 52 MiB; the
# first 50 varied prompts then add up to 250 more once (CUDA graphs and buffers of new lengths), and VRAM stays flat
# after that. Growth = 52 + 250, rounded up = 300 (G4 runs B1 and B3 on M2). The margin is on top of it.
$Growth = $P.Growth
# The driver keeps part of the card for itself ("Reserved" in nvidia-smi -q; 272 MiB on an RTX 4070 with Windows, 416 MiB
# on one with Linux). Past the rest, Windows moves memory to system RAM without an error and everything slows down (the
# "spill" below). Ceiling = the smaller of 97.7 % of the card and the card less the reserved part less 8 MiB.
# If nvidia-smi does not show the reserved part, 512 MiB is assumed. INKLING_CEILING_MIB overrides.
$ReservedMiB = $null; $InFb = $false
foreach ($l in @(& nvidia-smi -q -d MEMORY -i 0 2>$null)) {
    if ($l -match 'FB Memory Usage') { $InFb = $true }
    elseif ($InFb -and $l -match '^\s*Reserved\s*:\s*(\d+)\s*MiB') { $ReservedMiB = [int]$Matches[1]; break }
    elseif ($InFb -and $l -match 'Memory Usage') { break }
}
if ($null -eq $ReservedMiB) {
    $ReservedMiB = 512
    Write-Host "note   nvidia-smi does not show the card's reserved memory: 512 MiB assumed (INKLING_CEILING_MIB overrides)"
}
$Ceiling = [int](Pick 'CEILING_MIB' ([math]::Min([math]::Floor($TotalMiB * 0.977), $TotalMiB - $ReservedMiB - 8)))
# VRAM of the global K/V = cells x cell + staging. Staging = one global application's K and V of the cells in system
# RAM = (window - cells) x cell / 8. So cells = (budget - window x cell / 8) / (cell x 7 / 8), in steps of 256.
# The engine has one VRAM staging buffer for K and one for V, shared by all tiered caches and sized to one cache's host
#   tail: one global application's tail (cell / 8), as in Kintsugi's tier (one layer: cell / 16 there). Confirmed for
#   engine ace85a829; G4 load predicted within 15 MiB.
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
# Minimum 3,328 cells (Kintsugi's floor; the sliding-window caches are 2,048 cells, so they stay all in VRAM).
function CellsFor($fixed) {
    $budget = ([math]::Min($TotalMiB, $Ceiling) - $UsedMiB - $Margin - $fixed - $Growth) * 1MB - $Ctx * $Cell / $NGlobal
    [math]::Min($Ctx, [math]::Max(3328, [math]::Floor($budget / ($Cell * ($NGlobal - 1) / $NGlobal) / 256) * 256))
}
$Cells = [int](Pick 'CELLS' (CellsFor $Fixed))
if ($Cells -gt $Ctx) { $Cells = $Ctx }
# Window guard: with the tiered cache (cells in VRAM < window) the window is always a multiple of 32,768 tokens. Prompt
# reads past the VRAM line are wrong otherwise (a known bug of the tiered K/V). Any other window is rounded UP (and to
# at most 262,144 when it was not above it); with no tier (all cells in VRAM) any window is fine.
if ($Cells -lt $Ctx -and ($Ctx % 32768) -ne 0) {
    $CtxOld = $Ctx
    $Ctx = [int]([math]::Ceiling($Ctx / 32768) * 32768)
    if ($CtxOld -le 262144 -and $Ctx -gt 262144) { $Ctx = 262144 }
    Write-Host "note   window $CtxOld rounded up to $Ctx (a multiple of 32,768 with the tiered K/V cache)"
    $Cells = [int](Pick 'CELLS' (CellsFor $Fixed)); if ($Cells -gt $Ctx) { $Cells = $Ctx }
}
if ($Cells -le 3328) { Write-Host "note   only $Cells K/V cells fit in VRAM: decode reads most of the K/V cache from system RAM (close other programs that use the card)" }

# ---- Engine switches -------------------------------------------------------------------------------------------------
# No drafting in this release: the DSpark drafter of Inkling 2 Mini is not supported by this engine yet.
$env:GGML_CUDA_BATCH_INVARIANT = '1'          # a token's output does not depend on the batch it runs in
$env:LLAMA_ARG_CHECKPOINT_EVERY_NT = '8192'   # state checkpoints (the short-convolution state) for prompt-cache reuse
# The 40 sliding-window caches stay f16 whatever -ctk / -ctv say (they are small and fixed).
# It is an environment variable only (no CLI flag in engine ace85a829).
if (-not $env:LLAMA_INKLING_SWA_KV) { $env:LLAMA_INKLING_SWA_KV = 'f16' }
if (-not $env:GGML_CUDA_POOL_RESERVE_MIB) { $env:GGML_CUDA_POOL_RESERVE_MIB = '96' }
# Read path of the trellis experts for long prompts: the exact read only in this release (the int8 fast read is not
# tested on Inkling).
$env:GGML_ESCHA_PREFILL = 'recon'
# Spill guard: the engine prints a "spill" line when the server's dedicated GPU memory goes past this figure.
if (-not $env:GGML_CUDA_DEDICATED_CEILING_MIB) { $env:GGML_CUDA_DEDICATED_CEILING_MIB = "$Ceiling" }
# Thinking effort goes to the chat template; the template knows the 7 words and numbers from 0.0 to 0.99.
$env:LLAMA_ARG_CHAT_TEMPLATE_KWARGS = if ($EffortIsNum) { '{"reasoning_effort":' + $Effort + '}' } else { '{"reasoning_effort":"' + $Effort + '"}' }
# Any other value from a client (the template would stop on it: HTTP 500) becomes the fallback. The allow list compares
# text: a numeric INKLING_EFFORT is added so it passes as a number. The server sends the fallback as a JSON string, and
# the template stops on a string number ("0.5"), so the fallback is always a word: the one nearest the effort.
$EffortValue = @{ 'none' = 0.0; 'minimal' = 0.1; 'low' = 0.2; 'medium' = 0.7; 'high' = 0.9; 'xhigh' = 0.99; 'max' = 0.99 }
$EffortAllow = if ($EffortWords -contains $Effort) { $EffortWords } else { $EffortWords + @($Effort) }
$EffortFallback = $Effort
if (-not ($EffortWords -contains $Effort)) {   # nearest word; a tie goes to the lower effort (as start-server.sh)
    $EffortFallback = 'none'
    foreach ($w in @('minimal', 'low', 'medium', 'high', 'xhigh')) {
        if ([math]::Abs($EffortValue[$w] - $EffortNum) -lt [math]::Abs($EffortValue[$EffortFallback] - $EffortNum)) { $EffortFallback = $w }
    }
}
$env:PATH = "$Bin;" + $env:PATH          # the engine and the CUDA runtime DLLs are in bin\

$KeyFile = Pick 'KEY_FILE' ''
# Client address: the layer listens on it (or the server, with the layer off). The server behind the layer always
# listens on 127.0.0.1.
$ListenHost = Pick 'HOST' '127.0.0.1'
if ($ListenHost -ne '127.0.0.1' -and $ListenHost -ne 'localhost' -and -not $KeyFile) {
    Write-Host "WARNING listening on $ListenHost with NO API key: anyone who can reach this computer can use the server (and your GPU). Set INKLING_KEY_FILE to a file with a key."
}

# ---- Saved conversations: the layer (layer\bonsai_layer.py) runs in front of the server on the client port; the
# ---- server moves to 127.0.0.1:<port + 10000>. After each answer it saves a long conversation to disk (a full save,
# ---- then small deltas); after a restart, the next turn of a saved conversation reads only its new tokens.
# The layer's other tools (code interpreter, API cards and checks) are off in this build. It needs Python 3.10 or newer
# and nothing else (the standard library only). Without it the server runs plain, with a note.
$UseLayer = (Pick 'LAYER' '1') -eq '1'
# The layer serves one server slot (it saves and restores slot 0 and runs one request at a time): two slots run the
# plain server.
if ($UseLayer -and $Slots -gt 1) {
    Write-Host "note   $Slots slots: saved conversations are off (the layer serves one server slot); the plain server runs"
    $UseLayer = $false
}
if ($UseLayer) {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) {
        Write-Host "note   no python in PATH: saved conversations are off (the plain server runs). Install Python 3.10+ to use them, or set INKLING_LAYER=0 to hide this note."
        $UseLayer = $false
    } else {
        $ok = $false
        try { & python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>$null; $ok = $LASTEXITCODE -eq 0 } catch { }
        if (-not $ok) {
            Write-Host "note   saved conversations are off: they need Python 3.10 or newer (the python in PATH is older or does not run). The plain server runs."
            $UseLayer = $false
        }
    }
}
$ServerPort = $Port
$SlotDir = $null
if ($UseLayer) {
    $ServerPort = $Port + 10000
    $SlotDir = Pick 'SLOT_SAVE_PATH' (Join-Path $Root 'slots')
    if (-not $DryRun) { $SlotDir = (New-Item -ItemType Directory -Force -Path $SlotDir).FullName }
}
# Chat template: the model file's own (default), or a file.
$Template = Pick 'CHAT_TEMPLATE' 'model'
if ($Template -ne 'model' -and -not (Test-Path $Template)) { throw "chat template not found: $Template (set INKLING_CHAT_TEMPLATE)" }
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
$SrvArgs = @('-m', ('"' + $Model + '"'), '-ngl', '99', '-c', "$Ctx", '-fa', 'on', '-np', "$Slots", '-ub', "$Ub", '--fit', 'off', '--jinja',
          '-ctk', $KType, '-ctv', $VType, '--kv-vram-cells', "$Cells", '--kq-mask-packed', '--cache-ram', '4096', '--backend-sampling',
          '--temp', '1.0', '--top-p', '1.0', '--top-k', '0', '--min-p', '0',
          '--reasoning-budget', '20480', '-n', '24576', '--reasoning-budget-message', '"Now produce the complete answer."',
          '--reasoning-effort-allow', ($EffortAllow -join ','), '--reasoning-effort-fallback', $EffortFallback,
          '--reasoning-max-tokens-floor', '24576',
          '--host', $(if ($UseLayer) { '127.0.0.1' } else { $ListenHost }), '--port', "$ServerPort")
if ($Slots -gt 1) { $SrvArgs += '--kv-unified' }
if ($Template -ne 'model') { $SrvArgs += @('--chat-template-file', ('"' + $Template + '"')) }
if ($KeyFile) { $SrvArgs += @('--api-key-file', ('"' + $KeyFile + '"')) }
if ($SlotDir) { $SrvArgs += @('--slot-save-path', ('"' + $SlotDir + '"')) }
$Extra = Pick 'EXTRA' ''
if ($Extra) { $SrvArgs += $Extra.Split(' ') }

Write-Host $Product
Write-Host "model  $(Split-Path $Model -Leaf)"
Write-Host "listen $ListenHost`:$Port$(if ($UseLayer) { " (layer; llama-server on 127.0.0.1:$ServerPort)" })$(if ($KeyFile) { ", API key on" })"
Write-Host "kv     K $KType + V $VType (8 global caches), tiered: cells 0..$Cells in VRAM, $Cells..$Ctx in system RAM; 40 window caches f16 $SwaMiB MiB (free $($TotalMiB - $UsedMiB) MiB, reserved $ReservedMiB, ceiling $Ceiling, fixed $Fixed, growth $Growth, margin $Margin ($MarginWhy))$(if ($Slots -gt 1) { "; $Slots slots (unified cache)" })"
if ($Slots -gt 1) { Write-Host "note   $Slots slots share the $Ctx-token window (unified cache); long answers at temperature 0 are not bit-equal to one slot (near ties flip with the batch layout)" }
Write-Host "template $(if ($Template -eq 'model') { 'the model file''s own' } else { $Template })"
Write-Host "think  effort $Effort; sampling temperature 1.0, top-p 1.0, top-k 0, min-p 0; no drafting"
if ($DryRun) { Write-Host "kwargs $($env:LLAMA_ARG_CHAT_TEMPLATE_KWARGS)"; Write-Host ("args   " + ($SrvArgs -join ' ')); exit 0 }

$logs = Join-Path $Root 'logs'; New-Item -ItemType Directory -Force $logs | Out-Null
$log = Join-Path $logs "server-$Port.log"
$p = Start-Process -FilePath (Join-Path $Bin 'llama-server.exe') -ArgumentList ($SrvArgs -join ' ') -WorkingDirectory $Bin `
    -RedirectStandardOutput ($log + '.out') -RedirectStandardError $log -WindowStyle Hidden -PassThru
Write-Host "pid    $($p.Id)   log $log"
$lp = $null
$layerLog = Join-Path $logs "layer-$Port.log"
if ($UseLayer) {
    $env:BONSAI_SLOT_SAVE_PATH  = $SlotDir
    $env:BONSAI_SLOT_CACHE_GB   = Pick 'SLOT_CACHE_GB' '20'
    $env:BONSAI_SLOT_CACHE_AFTER = '1'
    if ($KeyFile) { $env:BONSAI_LAYER_KEY = (Get-Content -Raw $KeyFile).Trim() }
    # The stray-tag filter stays off: its rules are for Qwen-style tags, and the Inkling parser drops malformed tails.
    [string[]]$LayerFlags = @('--no-interpreter', '--no-cards', '--no-lint', '--no-example-check', '--no-finish-note', '--no-input-file', '--no-strip-stray-tags')
    $lp = Start-Process python -PassThru -WindowStyle Hidden -RedirectStandardOutput ($layerLog + '.out') -RedirectStandardError $layerLog `
        -ArgumentList (@(('"' + (Join-Path $Root 'layer\bonsai_layer.py') + '"'), '--host', $ListenHost, '--port', "$Port",
                         '--upstream', "http://127.0.0.1:$ServerPort") + $LayerFlags)
    Remove-Item Env:BONSAI_LAYER_KEY, Env:BONSAI_SLOT_SAVE_PATH, Env:BONSAI_SLOT_CACHE_GB, Env:BONSAI_SLOT_CACHE_AFTER -ErrorAction SilentlyContinue
    Write-Host "layer  pid $($lp.Id)   log $layerLog   clients use :$Port; saved conversations in $SlotDir"
}
Write-Host "stop   powershell -ExecutionPolicy Bypass -File `"$(Join-Path $Root 'stop.ps1')`""
if (-not $Wait) {
    # No wait for the load: the engine writes its GPU set-up to the log before the model load starts.
    for ($i = 0; $i -lt 60; $i++) {
        if ($p.HasExited -or (Select-String -Path $log -SimpleMatch -Pattern 'load_model: loading model' -Quiet -ErrorAction SilentlyContinue)) { break }
        Start-Sleep 1
    }
    $m = Get-NoGpuMessage $log; if ($m) { Stop-NoGpu $m }
}
if ($Wait) {
    for ($i = 0; $i -lt 150; $i++) {
        $up = $false
        try { $up = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://127.0.0.1:$Port/health").StatusCode -eq 200 } catch { }
        if ($up) {
            $m = Get-NoGpuMessage $log; if ($m) { Stop-NoGpu $m }
            # Warm-up, two requests in at most 600 s: (1) a chat request (about 1,100 tokens in, 8 out, no thinking);
            # (2) about 100 decoded tokens on the server port (/completion, about 200 tokens in): longer decodes take more
            # first-use memory (CUDA graphs, buffers; G4 saw +250 MiB in the first decodes after a chat-only warm-up).
            # The engine takes that memory now, so the VRAM printed below is the peak, and a shortage shows here.
            $text = (1..110 | ForEach-Object { "Entry $_ of the log has nothing to report." }) -join ' '
            $body = @{ model = 'warmup'; max_tokens = 8; temperature = 0; cache_prompt = $false
                       chat_template_kwargs = @{ reasoning_effort = 'none' }
                       messages = @(@{ role = 'user'; content = "$text Say OK." }) } | ConvertTo-Json -Depth 5
            $hdr = @{}
            if ($KeyFile) { $hdr['Authorization'] = 'Bearer ' + (Get-Content -Raw $KeyFile).Trim() }
            $sw = [Diagnostics.Stopwatch]::StartNew()
            try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 600 -Method Post -ContentType 'application/json' -Headers $hdr -Body $body "http://127.0.0.1:$Port/v1/chat/completions" | Out-Null }
            catch { Write-Host "note   the warm-up request failed: $($_.Exception.Message)" }
            $text2 = (1..20 | ForEach-Object { "Entry $_ of the log has nothing to report." }) -join ' '
            $body2 = @{ prompt = "$text2 Entry 21 of the log"; n_predict = 100; temperature = 0; cache_prompt = $false; ignore_eos = $true } | ConvertTo-Json
            $left = [math]::Max(30, [int](600 - $sw.Elapsed.TotalSeconds))
            try { Invoke-WebRequest -UseBasicParsing -TimeoutSec $left -Method Post -ContentType 'application/json' -Headers $hdr -Body $body2 "http://127.0.0.1:$ServerPort/completion" | Out-Null }
            catch { Write-Host "note   the second warm-up request (decode) failed: $($_.Exception.Message)" }
            $vram = [int]((& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) | Select-Object -First 1)
            # Spill check: the server's dedicated GPU memory against what the card holds. More in the process than on the
            # card means the driver moved the difference to system RAM (the engine prints "spill" lines for the same).
            $ded = 0
            try { $ded = [math]::Round(((Get-Counter "\GPU Process Memory(pid_$($p.Id)_*)\Dedicated Usage" -ErrorAction Stop).CounterSamples | Measure-Object CookedValue -Sum).Sum / 1MB) } catch { }
            $over = $ded - $vram
            $spillLines = @(Select-String -Path $log -Pattern 'memory_guard: spill' -ErrorAction SilentlyContinue).Count
            Write-Host "warm   $([math]::Round($sw.Elapsed.TotalSeconds, 1)) s; VRAM $vram MiB (ceiling $Ceiling), server process $ded MiB dedicated"
            if ($over -gt 64 -or $spillLines -gt 0) {
                Write-Host "SPILL  about $([math]::Max($over, 0)) MiB of the server's GPU memory are in system RAM: everything will be slow. Stop the server and start again with a larger INKLING_MARGIN (now $Margin) or fewer INKLING_CELLS (now $Cells)."
            }
            Write-Host "ready  http://127.0.0.1:$Port/v1"; exit 0
        }
        if ($p.HasExited) { $m = Get-NoGpuMessage $log; if ($m) { Stop-NoGpu $m }; Write-Host "server stopped during load: see $log"; exit 1 }
        if ($lp -and $lp.HasExited) { Write-Host "layer stopped: see $layerLog"; exit 1 }
        Start-Sleep 2
    }
    Write-Host "not ready after 300 s: see $log"; exit 1
}
