#!/usr/bin/env bash
# runs fair_perf (lean, dedicated loops for every variant) on our 2:4 and 4:8 simulators, in parallel.
# berkeley's original is measured separately by run_berkeley.sh with the identical dense loops.
set -o pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cy="$HOME/chipyard"
base="$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")"
source "$base/etc/profile.d/conda.sh"; conda deactivate 2>/dev/null || true
source "$cy/env.sh"
mkdir -p "$here/fair-logs"
tmp="$(mktemp -d)"   # make can't handle the ':' in this folder's name
cp "$here/bin/fair_perf_24-baremetal" "$here/bin/fair_perf_48-baremetal" "$tmp/"
run() {
  (cd "$cy/sims/verilator" && make CONFIG="$1" run-binary-fast LOADMEM=1 TIMEOUT_CYCLES=150000000 \
     BINARY="$tmp/$2-baremetal") > "$here/fair-logs/$2.log" 2>&1
  echo "$2 exit=$? $(date +%H:%M:%S)"
}
run GemminiNMRocketConfig fair_perf_24 &
run GemminiNM48RocketConfig fair_perf_48 &
wait
