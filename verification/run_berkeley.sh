#!/usr/bin/env bash
# runs berkeley_dense_perf on the UNMODIFIED berkeley gemmini simulator (built 2026-10-06 00:56 from
# stashed, stock sources). BREAK_SIM_PREREQ=1 stops make from rebuilding it out of the modified scala.
set -o pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cy="$HOME/chipyard"
base="$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")"
source "$base/etc/profile.d/conda.sh"; conda deactivate 2>/dev/null || true
source "$cy/env.sh"
mkdir -p "$here/berkeley-logs"
# make can't handle the ':' in this folder's name, so run a copy from a plain temp path
tmp="$(mktemp -d)"; cp "$here/bin/${PROG:-berkeley_dense_perf}-baremetal" "$tmp/"
(cd "$cy/sims/verilator" && make CONFIG=GemminiRocketConfig run-binary-fast BREAK_SIM_PREREQ=1 LOADMEM=1 \
   TIMEOUT_CYCLES=150000000 BINARY="$tmp/${PROG:-berkeley_dense_perf}-baremetal") > "$here/berkeley-logs/stock-${PROG:-berkeley_dense_perf}.log" 2>&1
echo "exit=$?"
