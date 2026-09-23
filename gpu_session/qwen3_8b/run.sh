#!/usr/bin/env bash
# Qwen3-8B, three ways, on one pod: df11pack DF11, df11pack idx8, and the
# official compressor. Verifies, hashes, compares, and writes report.json.
#
# Run from the repo root on the pod:  bash gpu_session/qwen3_8b/run.sh
# Everything lands in /workspace (the persistent volume). A watchdog stops the
# pod after 12 hours whatever happens, so a forgotten pod cannot drain credit.
set -uo pipefail

W=/workspace
REPO=$(pwd)
MODEL_REV=b968826d9c46dd6066d109eabc6255188de91218
LOG=$W/run.log
mkdir -p $W/out
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
step() { # name, command...: runs under GNU time, records wall/RSS/exit.
    local name=$1; shift
    log "START $name"
    /usr/bin/time -v -o "$W/out/$name.time" "$@" > "$W/out/$name.log" 2>&1
    local rc=$?
    echo "$rc" > "$W/out/$name.rc"
    log "END   $name rc=$rc"
    return $rc
}

if [ -n "${RUNPOD_POD_ID:-}" ]; then
    ( sleep 43200; echo "watchdog: stopping pod" >> "$LOG"; runpodctl stop pod "$RUNPOD_POD_ID" ) &
fi

log "machine: $(nproc) cpus, $(free -g | awk '/Mem:/{print $2}') GB RAM, $(df -h $W | awk 'NR==2{print $4}') free"

# 1. Environment: the exact local versions; CPU torch.
if [ ! -x $W/venv/bin/python ]; then
    apt-get update -qq && apt-get install -y -qq time > /dev/null
    python3 -m venv $W/venv
    $W/venv/bin/pip install -q --upgrade pip
    $W/venv/bin/pip install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
    $W/venv/bin/pip install -q -r gpu_session/qwen3_8b/requirements.txt
fi
$W/venv/bin/pip freeze > $W/out/pip_freeze.txt

if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /dev/null
fi
step build "$HOME/.cargo/bin/cargo" build --release -p df11pack || exit 1
BIN=$REPO/target/release/df11pack

# 2. The source model, pinned.
if [ ! -f $W/src/config.json ]; then
    step download $W/venv/bin/hf download Qwen/Qwen3-8B --revision $MODEL_REV --local-dir $W/src || exit 1
fi

# 3. df11pack, standard DF11.
rm -rf $W/out/df11pack
step df11pack $BIN compress $W/src --arch qwen3-8b -o $W/out/df11pack
step df11pack_verify $BIN verify $W/out/df11pack --source $W/src --arch qwen3-8b --level full

# 4. df11pack, idx8 (not DF11), every block verified before writing.
rm -rf $W/out/idx8
step idx8 $BIN compress $W/src --arch qwen3-8b -o $W/out/idx8 --index idx8 --idx8-block 64 --safe

# 5. The official compressor, same pattern as the official release.
rm -rf $W/out/official
step official $W/venv/bin/python phase0/run_official.py --model $W/src --out $W/out/official --pattern mainstream

# 6. Hashes, sizes, comparisons.
step report $W/venv/bin/python gpu_session/qwen3_8b/report.py
log "ALL DONE"
touch $W/out/DONE
