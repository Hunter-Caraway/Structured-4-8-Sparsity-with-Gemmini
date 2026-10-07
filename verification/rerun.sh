#!/usr/bin/env bash
# re-runs the six rtl benchmark tests on the existing nm24 / nm48 simulators, logs only.
# does not touch results/cycles.csv (run-flow.sh bench would append to it)
set -o pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$here/rerun-logs"
cy="$HOME/chipyard"
base="$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")"
source "$base/etc/profile.d/conda.sh"; conda deactivate 2>/dev/null || true
source "$cy/env.sh"
bm="$cy/generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC"
run_cfg() {
  local cfg="$1" tag="$2" sfx="$3" t
  for t in nm_sparse_debug nm_sparse_matmul nm_sparse_perf; do
    (cd "$cy/sims/verilator" && make CONFIG="$cfg" run-binary-fast BINARY="$bm/${t}${sfx}-baremetal" \
      LOADMEM=1 TIMEOUT_CYCLES=150000000) > "$out/$tag-$t.log" 2>&1
    echo "$tag $t exit=$? $(date +%H:%M:%S)" >> "$out/status.txt"
  done
}
: > "$out/status.txt"
echo "start $(date)" >> "$out/status.txt"
run_cfg GemminiNMRocketConfig nm24 "" &
run_cfg GemminiNM48RocketConfig nm48 _48 &
wait
echo "done $(date)" >> "$out/status.txt"
