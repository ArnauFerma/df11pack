#!/usr/bin/env bash
# Byte comparison of df11pack against a published ComfyUI DF11 release (mingyi456's), CPU only:
# download the source file and the release file (pinned revisions), compress the source with
# df11pack, then compare every tensor byte for byte (verify_repack.py) and the files by SHA-256.
#   check_comfyui_release.sh NAME SRC_REPO SRC_REV SRC_FILE REL_REPO REL_REV REL_FILE ARCH [WORKDIR]
# Output: phase0/results/comfyui/NAME.txt (and the df11pack log next to it).
set -uo pipefail
NAME=$1 SRC_REPO=$2 SRC_REV=$3 SRC_FILE=$4 REL_REPO=$5 REL_REV=$6 REL_FILE=$7 ARCH=$8 WD=${9:-/tmp/comfyui_check}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE")
OUT=$HERE/results/comfyui; mkdir -p $OUT $WD
PY=${PY:-python3}; BIN=${BIN:-$ROOT/target/release/df11pack}
R=$OUT/$NAME.txt
{
echo "# $(date -u +%FT%TZ) df11pack $($BIN --version 2>/dev/null) ($(git -C $ROOT rev-parse --short HEAD)), arch $ARCH"
echo "# source  $SRC_REPO@$SRC_REV $SRC_FILE"
echo "# release $REL_REPO@$REL_REV $REL_FILE"
echo "# machine: $(lscpu | sed -n 's/^Model name: *//p' | head -1), $(free -g | awk '/^Mem/{print $2}') GB RAM, $(uname -sr)"
} > $R
dl() { $PY -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('$1', '$3', revision='$2', local_dir='$WD/$4'))" 2>&1 | tail -1; }
SRC=$(dl $SRC_REPO $SRC_REV $SRC_FILE src); REL=$(dl $REL_REPO $REL_REV $REL_FILE rel)
[ -f "$SRC" ] && [ -f "$REL" ] || { echo "FAILED download: $SRC | $REL" | tee -a $R; exit 1; }
rm -rf $WD/out
TIME=""; [ -x /usr/bin/time ] && TIME="/usr/bin/time -v"   # peak memory when GNU time exists
T0=$(date +%s)
$TIME $BIN compress "$SRC" --arch $ARCH -o $WD/out --ram ${RAM:-2G} > $OUT/$NAME.df11pack.log 2>&1
RC=$?
echo "df11pack exit $RC after $(( $(date +%s) - T0 )) s (log: $NAME.df11pack.log)" >> $R
[ $RC -eq 0 ] || { echo "FAILED: df11pack did not finish; no comparison" >> $R
    sed -i "s#$WD/#<workdir>/#g; s#$ROOT/#<df11pack>/#g" $R $OUT/$NAME.df11pack.log; cat $R; exit 1; }
echo "## per tensor: release (left) against df11pack output (right)" >> $R
$PY $HERE/verify_repack.py "$REL" $WD/out >> $R 2>&1
echo "## files (SHA-256)" >> $R
sha256sum "$REL" $WD/out/*.safetensors | sed "s#$WD/##" >> $R
sed -i "s#$WD/#<workdir>/#g; s#$ROOT/#<df11pack>/#g" $R $OUT/$NAME.df11pack.log   # no local paths in the repo
cat $R
