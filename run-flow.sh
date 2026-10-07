#!/usr/bin/env bash
# run-flow.sh: the whole Gemmini N:M (2:4 / 4:8) sparsity flow, from environment check to verified figures.
#
# usage: ./run-flow.sh <step> [args]          (./run-flow.sh with no step lists them)
#
#  setup
#    env               print the commands to set up a fresh terminal by hand
#    check             tools, simulators, branches, python libraries
#  build
#    spike-lib         build + install the Gemmini Spike extension (libgemmini)
#    tests             build gemmini-rocc-tests, plus the _48 (4:8) sparse variants
#    fair-build        build the fair benchmark into verification/build/ and compare it with the recorded programs
#    sim <cfg>         build a Verilator simulator. cfg = nm24 | nm48 | dense (see the note on dense below)
#  simulate (slow; sleep is blocked while these run)
#    spike             quick functional tests on Spike, including the fair benchmark (seconds)
#    fair [which]      fair benchmark on RTL. which = all (default) | nm24 | nm48 | berkeley | sameprog
#                      results go to verification/runs/<stamp>/ and are compared with the recorded ones
#    rerun             the original six RTL tests again, into verification/runs/<stamp>/, then compared
#    baseline          stock matmul-baremetal on Berkeley's simulator (very slow: full instruction trace)
#    bench <cfg>       ORIGINAL benchmark (nm_sparse_perf; superseded loop) on nm24|nm48 -> results/cycles.csv
#  check and report (python; seconds)
#    verify            everything: verification/verify_all.py (writes verification/verify_all_report.md)
#    compare <dir>     compare a run folder with the recorded results
#    math              independent math check (sparse math == dense math, fault injection)
#    patterns          how much of the weights 2:4, 4:8 and NVIDIA's paired 4:8 keep
#    figures           the paper's numbers + chart, and the advanced figures (analysis/advanced/)
#    checks            math + patterns + verify + figures
#  pipelines
#    all               spike-lib, tests, fair-build, spike, sim nm24, sim nm48, fair all, checks
#
# environment variables
#   CHIPYARD_DIR    where chipyard lives (default ~/chipyard; no spaces or ':' allowed)
#   JOBS            make -j for simulator builds (default: picked from free RAM, ~2.5 GB per job)
#   TIMEOUT_CYCLES  simulation cycle limit (default 150M; chipyard's own 10M is too short)
#   FAIR_BIN        folder with the fair benchmark programs (default verification/bin = the recorded ones;
#                   set FAIR_BIN=verification/build to time freshly built programs)
#   PYTHON          python to use (default: chipyard's conda env, which has numpy/pandas/matplotlib)
#   FORCE=1         allow `sim dense` to replace Berkeley's stock simulator (see below)
#
# notes learned the hard way
#   * Berkeley's stock simulator (GemminiRocketConfig) was built from UNMODIFIED Gemmini sources. Chipyard
#     rebuilds a simulator whenever the Scala changes, so every run on it uses BREAK_SIM_PREREQ=1, and
#     `sim dense` refuses to rebuild it (from the sparse branch) unless FORCE=1.
#   * make can't take paths containing ':' (this folder has one), so binaries are copied to a temp folder first.
#   * the laptop suspends on lid close / idle, which pauses simulations; long steps hold a sleep inhibitor.

set -eo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cy="${CHIPYARD_DIR:-$HOME/chipyard}"
gem="$cy/generators/gemmini"
tests_dir="$gem/software/gemmini-rocc-tests"
bm="$tests_dir/build/bareMetalC"
ver="$here/verification"
logs="$here/logs"
results="$here/results"
py="${PYTHON:-$cy/.conda-env/bin/python}"
fair_bin="${FAIR_BIN:-$ver/bin}"
[[ "$fair_bin" = /* ]] || fair_bin="$here/$fair_bin"
stamp="$(date +%Y%m%d-%H%M%S)"

say()  { printf '\n== %s\n' "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
cfg_name() {
  case "$1" in
    dense) echo GemminiRocketConfig ;;
    nm24)  echo GemminiNMRocketConfig ;;
    nm48)  echo GemminiNM48RocketConfig ;;
    *) die "unknown config '$1' (want nm24, nm48 or dense)" ;;
  esac
}

case "$cy" in *" "*|*:*) die "chipyard path '$cy' has a space or ':' in it; it has to live somewhere plain" ;; esac
[ -f "$cy/env.sh" ] || die "no env.sh in $cy; did build-setup.sh run? (see chipyard-setup-notes.md)"

# ---------------------------------------------------------------- helpers
# fedora / rhel: conda deactivate first, then env.sh (the chipyard docs say so)
load_env() {
  local base
  base="$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")"
  # shellcheck disable=SC1091
  source "$base/etc/profile.d/conda.sh"
  conda deactivate 2>/dev/null || true
  # shellcheck disable=SC1091
  source "$cy/env.sh"
  [ -n "$RISCV" ] || die "RISCV not set after sourcing env.sh"
  export LD_LIBRARY_PATH="$RISCV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

need_py() { [ -x "$py" ] || die "no python at $py (set PYTHON=...)"; }

# verilator compiles take ~2.5 GB each on gemmini
pick_jobs() {
  if [ -n "$JOBS" ]; then echo "$JOBS"; return; fi
  local avail_kb jobs
  avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  jobs=$(( avail_kb / (2560 * 1024) ))
  [ "$jobs" -lt 1 ] && jobs=1
  [ "$jobs" -gt "$(nproc)" ] && jobs=$(nproc)
  echo "$jobs"
}

# one cleanup for everything this script starts: the sleep inhibitor and the temp binary folder
inhibitor=""
tmpbin=""
cleanup() {
  [ -n "$inhibitor" ] && kill "$inhibitor" 2>/dev/null
  [ -n "$tmpbin" ] && rm -rf "$tmpbin"
  return 0
}
trap cleanup EXIT

# hold off suspend/idle sleep until this script exits (a suspend pauses the simulation)
stay_awake() {
  command -v systemd-inhibit >/dev/null || return 0
  [ -n "$inhibitor" ] && return 0
  systemd-inhibit --what=sleep:idle --who="run-flow.sh" --why="$1" sleep infinity &
  inhibitor=$!
  echo "sleep blocked while this runs (closing the lid may still suspend, depending on the lid setting)"
}

# make can't handle ':' in BINARY=, so binaries are staged in a plain temp folder.
# make_tmpbin must run in the main shell; stage_bin is then safe inside $( )
make_tmpbin() { [ -n "$tmpbin" ] || tmpbin="$(mktemp -d)"; }
stage_bin() {
  [ -n "$tmpbin" ] || die "internal: make_tmpbin first"
  cp "$1" "$tmpbin/" && echo "$tmpbin/$(basename "$1")"
}

# run one binary on one simulator: sim_run <cfg> <binary> <log> [extra make args...]
sim_run() {
  local cfg="$1" bin="$2" log="$3"; shift 3
  (cd "$cy/sims/verilator" && make CONFIG="$cfg" run-binary-fast LOADMEM=1 \
     TIMEOUT_CYCLES="${TIMEOUT_CYCLES:-150000000}" BINARY="$bin" "$@") > "$log" 2>&1
}

# compile a bare-metal test the same way gemmini-rocc-tests does: rv_cc <src> <out> [-Dflags...]
rv_cc() {
  local src="$1" out="$2" r="$tests_dir"; shift 2
  riscv64-unknown-elf-gcc "$@" -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
    -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc \
    -Wa,-march=rv64gc -lm -lgcc -I"$r/riscv-tests" -I"$r/riscv-tests/env" -I"$r" \
    -I"$r/riscv-tests/benchmarks/common" -DID_STRING= -DPRINT_TILE=0 -nostdlib -nostartfiles -static \
    -T "$r/riscv-tests/benchmarks/common/test.ld" -DBAREMETAL=1 "$src" -o "$out" \
    "$r/riscv-tests/benchmarks/common/syscalls.c" "$r/riscv-tests/benchmarks/common/crt.S" 2>&1 | grep -v RWX || true
  [ -s "$out" ] || die "compile failed: $src"
}

# ---------------------------------------------------------------- setup
step_env() {
  cat <<EOF
# fresh terminal setup (fedora needs the deactivate first)
conda deactivate
cd $cy
source env.sh
EOF
}

step_check() {
  load_env
  say "tools"
  echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
  echo "RISCV=$RISCV"
  for t in spike verilator firtool riscv64-unknown-elf-gcc; do
    printf '  %-24s %s\n' "$t" "$(command -v "$t" || echo MISSING)"
  done
  [ -f "$RISCV/lib/libgemmini.so" ] && echo "  libgemmini.so installed" || echo "  libgemmini.so MISSING (run spike-lib)"
  say "simulators"
  local c s
  for c in dense nm24 nm48; do
    s="$cy/sims/verilator/simulator-chipyard.harness-$(cfg_name $c)"
    if [ -x "$s" ]; then printf '  %-6s %s  (built %s)\n' "$c" "$(basename "$s")" "$(date -r "$s" '+%Y-%m-%d %H:%M')"
    else printf '  %-6s MISSING (run: ./run-flow.sh sim %s)\n' "$c" "$c"; fi
  done
  say "code branches"
  for r in "$gem" "$gem/software/libgemmini" "$tests_dir"; do
    printf '  %-20s %s @ %s%s\n' "$(basename "$r")" "$(git -C "$r" branch --show-current)" \
      "$(git -C "$r" rev-parse --short HEAD)" "$(git -C "$r" diff --quiet && echo '' || echo '  (uncommitted changes)')"
  done
  say "python ($py)"
  if [ -x "$py" ]; then "$py" -c "import numpy, pandas, matplotlib; print('  numpy', numpy.__version__, '| pandas', pandas.__version__, '| matplotlib', matplotlib.__version__)"
  else echo "  MISSING"; fi
  say "disk"
  df -h "$HOME" | tail -1
}

# ---------------------------------------------------------------- build
step_spike_lib() {
  load_env
  say "building libgemmini (spike extension)"
  make -C "$gem/software/libgemmini" install 2>&1 | tee "$logs/runs/libgemmini-$stamp.log" | grep -E "error|cp " || true
  [ -f "$RISCV/lib/libgemmini.so" ] || die "libgemmini.so didn't install"
}

# 4:8 versions of the sparse tests, built with explicit -D flags so they don't depend on
# whichever config was elaborated last
build_48_tests() {
  local t
  for t in nm_sparse_debug nm_sparse_matmul nm_sparse_perf; do
    rv_cc "$tests_dir/bareMetalC/$t.c" "$bm/${t}_48-baremetal" -DNM_N=4 -DNM_M=8
  done
}

step_tests() {
  load_env
  say "building gemmini-rocc-tests"
  # every elaboration rewrites gemmini_params.h (an n:m config adds NM_N/NM_M); put the committed one back
  git -C "$tests_dir" checkout -- include/gemmini_params.h
  # a killed make can leave 0-byte binaries that look up to date; clear them first
  [ -d "$tests_dir/build" ] && find "$tests_dir/build" -type f -size 0 -name '*-baremetal' -delete
  (cd "$tests_dir" && ./build.sh) > "$logs/runs/rocc-tests-$stamp.log" 2>&1 || die "test build failed, see $logs/runs/rocc-tests-$stamp.log"
  build_48_tests
  local built=("$bm"/*-baremetal)
  echo "${#built[@]} bare-metal tests built (including the _48 sparse variants)"
}

step_fair_build() {
  load_env
  local out="$ver/build" n=0 same=0 b
  mkdir -p "$out"
  say "building the fair benchmark into ${out#"$here"/}"
  rv_cc "$ver/fair_perf.c" "$out/fair_perf_24-baremetal" -DNM_N=2 -DNM_M=4
  rv_cc "$ver/fair_perf.c" "$out/fair_perf_48-baremetal" -DNM_N=4 -DNM_M=8
  rv_cc "$ver/fair_perf.c" "$out/fair_perf_dense_only-baremetal" -DDENSE_ONLY -DNM_N=2 -DNM_M=4
  # compare what the chip actually loads (code + data). the raw files always differ in a few symbol-table
  # bytes, because gcc records random temp-file names there
  loadimg() {
    local img; img="$(mktemp)"
    riscv64-unknown-elf-objcopy -O binary "$1" "$img" || die "objcopy failed on $1"
    sha256sum "$img" | cut -c1-16; rm -f "$img"
  }
  for b in "$out"/*-baremetal; do
    n=$((n + 1))
    if [ "$(loadimg "$b")" = "$(loadimg "$ver/bin/$(basename "$b")")" ]; then
      same=$((same + 1)); echo "  identical code and data to the recorded program: $(basename "$b")"
    else
      echo "  DIFFERS from the recorded program: $(basename "$b") (timings may differ; FAIR_BIN=verification/build to time it)"
    fi
  done
  echo "$same of $n programs match the ones that produced the recorded results"
}

step_sim() {
  local which="${1:-}" cfg jobs log s
  [ -n "$which" ] || die "usage: ./run-flow.sh sim nm24|nm48|dense"
  cfg="$(cfg_name "$which")"
  s="$cy/sims/verilator/simulator-chipyard.harness-$cfg"
  if [ "$which" = dense ] && [ -x "$s" ] && [ "${FORCE:-0}" != 1 ]; then
    die "$cfg already exists and was built from Berkeley's UNMODIFIED sources. Rebuilding now would build it from
the sparse-nm branch instead (sparsity off, not byte-for-byte Berkeley's). Use FORCE=1 if that's what you want."
  fi
  load_env
  jobs="$(pick_jobs)"
  log="$logs/runs/sim-$cfg-$stamp.log"
  stay_awake "building $cfg"
  say "building verilator sim $cfg with -j$jobs -> ${log#"$here"/}"
  if ! (cd "$cy/sims/verilator" && make -j"$jobs" CONFIG="$cfg") > "$log" 2>&1; then
    # the OOM killer shows up as a killed compiler; retry slower once
    if grep -qiE "killed|signal 9|out of memory" "$log" && [ "$jobs" -gt 2 ]; then
      say "looks like it ran out of memory, retrying with -j2"
      (cd "$cy/sims/verilator" && make -j2 CONFIG="$cfg") >> "$log" 2>&1 || die "sim build failed, see $log"
    else
      die "sim build failed, see $log"
    fi
  fi
  ls -la "$s"
}

# ---------------------------------------------------------------- simulate
step_spike() {
  load_env
  local log="$logs/runs/spike-$stamp.log" fails=0 t
  say "spike tests -> ${log#"$here"/}"
  for t in "$bm/mvin_mvout-baremetal" "$bm/matmul-baremetal" "$bm/nm_sparse_sw-baremetal" \
           "$bm/nm_sparse_matmul-baremetal" "$fair_bin/fair_perf_24-baremetal" "$fair_bin/fair_perf_dense_only-baremetal"; do
    if [ ! -f "$t" ]; then echo "skip  $(basename "$t" -baremetal) (not built)"; continue; fi
    if spike --extension=gemmini "$t" >> "$log" 2>&1; then echo "pass  $(basename "$t" -baremetal)"
    else echo "FAIL  $(basename "$t" -baremetal)"; fails=$((fails + 1)); fi
  done
  echo "(spike's sparse model is 2:4 by default, so the 4:8 programs are checked on the RTL instead)"
  [ "$fails" -eq 0 ] || die "$fails spike test(s) failed"
}

step_fair() {
  local which="${1:-all}" run="$ver/runs/fair-$stamp" fails=0 job pids=() names=()
  case "$which" in all|nm24|nm48|berkeley|sameprog) ;; *) die "fair: want all|nm24|nm48|berkeley|sameprog" ;; esac
  load_env; need_py
  mkdir -p "$run"
  make_tmpbin
  stay_awake "fair benchmark simulations"
  say "fair benchmark ($which) -> ${run#"$here"/}   (each takes ~15-40 min; they run in parallel)"
  launch() {   # launch <name> <cfg> <program> [extra make args]
    local name="$1" cfg="$2" prog="$3" bin; shift 3
    [ -f "$fair_bin/$prog-baremetal" ] || die "missing $fair_bin/$prog-baremetal (./run-flow.sh fair-build)"
    bin="$(stage_bin "$fair_bin/$prog-baremetal")"
    ( sim_run "$cfg" "$bin" "$run/$name.log" "$@"; echo "  $name finished $(date +%H:%M:%S) (exit $?)" ) &
    pids+=("$!"); names+=("$name")
    echo "  started $name on $cfg"
  }
  [[ "$which" =~ ^(all|nm24)$ ]]     && launch nm24 GemminiNMRocketConfig fair_perf_24
  [[ "$which" =~ ^(all|nm48)$ ]]     && launch nm48 GemminiNM48RocketConfig fair_perf_48
  [[ "$which" =~ ^(all|berkeley)$ ]] && launch berkeley GemminiRocketConfig fair_perf_dense_only BREAK_SIM_PREREQ=1
  [[ "$which" =~ ^(all|sameprog)$ ]] && launch nm24_sameprog GemminiNMRocketConfig fair_perf_dense_only
  for job in "${pids[@]}"; do wait "$job" || fails=$((fails + 1)); done
  say "comparing with the recorded results"
  "$py" "$ver/compare_runs.py" "$run" || die "the new run differs from the recorded results (see above)"
}

step_rerun() {
  local run="$ver/runs/rerun-$stamp" pids=() c t sfx
  load_env; need_py
  mkdir -p "$run"
  stay_awake "re-running the original RTL tests"
  say "original six RTL tests -> ${run#"$here"/}   (nm24 and nm48 in parallel)"
  for c in nm24 nm48; do
    sfx=""; [ "$c" = nm48 ] && sfx="_48"
    (
      for t in nm_sparse_debug nm_sparse_matmul nm_sparse_perf; do
        sim_run "$(cfg_name $c)" "$bm/${t}${sfx}-baremetal" "$run/orig-$c-$t.log" || true
        echo "  $c $t finished $(date +%H:%M:%S)"
      done
    ) &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p"; done
  say "comparing with the recorded results"
  "$py" "$ver/compare_runs.py" "$run" || die "the re-run differs from the recorded results (see above)"
}

step_baseline() {
  load_env
  local log="$logs/baseline/baseline_matmul-$stamp.log"
  mkdir -p "$logs/baseline"
  stay_awake "stock baseline simulation"
  say "stock baseline: matmul-baremetal on Berkeley's GemminiRocketConfig -> ${log#"$here"/}"
  # run-binary (not -fast) logs every instruction: very slow, GB-sized trace in sims/verilator/output
  (cd "$cy/sims/verilator" && make CONFIG=GemminiRocketConfig run-binary BREAK_SIM_PREREQ=1 \
     BINARY="$bm/matmul-baremetal" TIMEOUT_CYCLES="${TIMEOUT_CYCLES:-150000000}") 2>&1 | tee "$log"
  grep -E "PASSED|FAILED|Completed after" "$log" || true
}

csv_row() {
  [ -f "$results/cycles.csv" ] || echo "timestamp,config,test,variant,status,total_cycles,array_busy_cycles,tiles" > "$results/cycles.csv"
  echo "$stamp,$1,$2,$3,$4,$5,$6,$7" >> "$results/cycles.csv"
}

# the ORIGINAL benchmark. its generic loop slowed dense mode more than sparse mode, so its whole-job
# speed-ups are superseded by `fair`. kept to reproduce results/cycles.csv
step_bench() {
  local which="${1:-nm24}" cfg sfx="" fails=0 t log status
  cfg="$(cfg_name "$which")"
  [ "$which" = dense ] && die "bench needs a sparse config (nm24 or nm48)"
  [ "$which" = nm48 ] && sfx="_48"
  load_env
  stay_awake "original benchmark"
  echo "note: this is the original benchmark (superseded loop). For numbers to cite, use: ./run-flow.sh fair"
  for t in nm_sparse_debug nm_sparse_matmul nm_sparse_perf; do
    log="$logs/runs/bench-$which-$t-$stamp.log"
    say "$t on $cfg -> ${log#"$here"/}"
    [ -f "$bm/${t}${sfx}-baremetal" ] || die "missing $bm/${t}${sfx}-baremetal, run ./run-flow.sh tests"
    sim_run "$cfg" "$bm/${t}${sfx}-baremetal" "$log" || true
    status=FAILED; grep -q "$t passed" "$log" && status=passed
    [ "$status" = passed ] || fails=$((fails + 1))
    echo "$status  $t"
    case "$t" in
      nm_sparse_matmul)
        csv_row "$cfg" "$t" dense "$status" "$(grep -oP 'dense\s+cycles: \K[0-9]+' "$log" | tail -1)" "" ""
        csv_row "$cfg" "$t" sparse "$status" "$(grep -oP 'sparse cycles: \K[0-9]+' "$log" | tail -1)" "" "" ;;
      nm_sparse_perf)
        grep -E '^[a-z_]+ +total' "$log" | while read -r name _ total _ _ _ tiles _ _ rest; do
          busy=$(echo "$rest" | grep -oP 'ex_active=\K[0-9]+')
          csv_row "$cfg" "$t" "$name" "$status" "$total" "$busy" "$tiles"
          printf '   %-16s total %6s  array busy %6s  tiles %s\n' "$name" "$total" "$busy" "$tiles"
        done ;;
    esac
  done
  say "results appended to results/cycles.csv"
  [ "$fails" -eq 0 ] || die "$fails sparse test(s) failed on $cfg"
}

# ---------------------------------------------------------------- check and report (python)
step_verify()   { need_py; say "verify_all.py"; "$py" "$ver/verify_all.py" "$@"; }
step_compare()  { need_py; [ -n "${1:-}" ] || die "usage: ./run-flow.sh compare verification/runs/<stamp>"; "$py" "$ver/compare_runs.py" "$1"; }
step_math()     { need_py; say "independent math check"; "$py" "$ver/independent_math_check.py"; }
step_patterns() { need_py; say "pattern comparison"; "$py" "$ver/pattern_fidelity.py"; }
step_figures() {
  need_py
  say "paper numbers + chart (verification/paper_data.json, paper_chart.png)"
  "$py" "$ver/paper_data.py" > /dev/null && echo "  written"
  say "advanced figures (analysis/advanced/)"
  "$py" "$here/analysis/advanced_visuals.py"
}
step_checks() { step_math; step_patterns; step_verify --skip-patches; step_figures; }

# ---------------------------------------------------------------- dispatch
step="${1:-}"
shift || true
case "$step" in
  env)        step_env ;;
  check)      step_check ;;
  spike-lib)  step_spike_lib ;;
  tests)      step_tests ;;
  fair-build) step_fair_build ;;
  sim)        step_sim "$@" ;;
  spike)      step_spike ;;
  fair)       step_fair "$@" ;;
  rerun)      step_rerun ;;
  baseline)   step_baseline ;;
  bench)      step_bench "$@" ;;
  verify)     step_verify "$@" ;;
  compare)    step_compare "$@" ;;
  math)       step_math ;;
  patterns)   step_patterns ;;
  figures)    step_figures ;;
  checks)     step_checks ;;
  all)
    step_spike_lib
    step_tests
    step_fair_build
    step_spike
    step_sim nm24
    step_sim nm48
    step_fair all
    step_checks
    ;;
  ""|-h|--help|help) sed -n '2,/^set -eo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown step '$step'"; sed -n '4,30p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
