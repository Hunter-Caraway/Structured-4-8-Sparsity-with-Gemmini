// fair benchmark: every variant gets its own lean, dedicated issue loop, written the same way as
// berkeley_dense_perf.c (which ran on the unmodified berkeley gemmini). nm_sparse_perf.c used one
// generic loop for dense and sparse, and its runtime branches made the processor slower, so its
// weight-reuse numbers understated a well-written dense loop.
// same matrices as nm_sparse_perf / berkeley_dense_perf (same rand() sequence), pruned NM_N:NM_M.
// every real run is checked against the cpu reference (2048 values).
// build twice: -DNM_N=2 -DNM_M=4 and -DNM_N=4 -DNM_M=8. runs on the GemminiNM*RocketConfig sims.
// -DDENSE_ONLY builds the same file without the sparse gemmini runs, for berkeley's unmodified gemmini
// (which has no sparse mode), so both chips run byte-for-byte the same dense loop source.

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/gemmini_counter.h"
#include "include/gemmini_nm.h"

#define NM_I (4*DIM)
#define NM_K (8*DIM)
#define NM_J (2*DIM)
#define IT (NM_I / DIM)
#define KT (NM_K / DIM)
#define JT (NM_J / DIM)
#define KC (NM_K / NM_M * NM_N)
#define KCT (KC / DIM)

static elem_t a[NM_I][NM_K] row_align(1);
static elem_t w[NM_K][NM_J] row_align(1);
static elem_t wc[KC][NM_J] row_align(1);
static nm_idx_t idx[KC][NM_J];
static uint64_t meta[KCT][JT][2 * DIM];
static acc_t c_out[NM_I][NM_J] row_align_acc(1);
static acc_t gold[NM_I][NM_J];

#define A_SP(it, kt) (((kt) % 2) * BANK_ROWS + ((it) * (KT / 2) + (kt) / 2) * DIM)
#define W_SP(kt, jt) (2 * BANK_ROWS + ((kt) * JT + (jt)) * DIM)
#define WC_SP(t, jt) (3 * BANK_ROWS + ((t) * JT + (jt)) * DIM)
#define ACC_BIT ((uint32_t)1 << (ADDR_LEN - 1))
#define ACCUM_BIT ((uint32_t)1 << (ADDR_LEN - 2))
#define FULL_BIT ((uint32_t)1 << (ADDR_LEN - 3))
#define C_ACC(it, jt) (((it) * JT + (jt)) * DIM)

// identical to berkeley_dense_perf.c
static int cpu_only = 0;
#define SINK(x, y) asm volatile ("" :: "r"((uint64_t)(x)), "r"((uint64_t)(y)))
#define PRELOAD(b, c) do { if (cpu_only) SINK(b, c); else { gemmini_preload(b, c); } } while (0)
#define COMPUTE_PRE(a, bd) do { if (cpu_only) SINK(a, bd); else { gemmini_compute_preloaded(a, bd); } } while (0)
#define COMPUTE_ACC(a, bd) do { if (cpu_only) SINK(a, bd); else { gemmini_compute_accumulated(a, bd); } } while (0)
static inline void send_meta(const uint64_t * packed) {
  if (!cpu_only) { nm_send_meta(packed, NM_M); return; }
  for (int i = 0; i < nm_meta_cmds(NM_M); i++) SINK(packed[2 * i], packed[2 * i + 1]);
}

static const int ctr_ids[8] = {
  EXE_ACTIVE_CYCLE, RESERVATION_STATION_FULL_CYCLES, EXE_CONTROL_Q_BLOCK_CYCLE, EXE_OVERLAP_HAZ_CYCLE,
  SCRATCHPAD_A_WAIT_CYCLE, SCRATCHPAD_B_WAIT_CYCLE, SCRATCHPAD_D_WAIT_CYCLE, MAIN_EX_CYCLES
};
static const char * ctr_names[8] = {
  "ex_active", "rs_full", "ctrl_q_block", "overlap_haz",
  "spad_a_wait", "spad_b_wait", "spad_d_wait", "main_ex"
};
static uint32_t ctr_vals[8];
static void ctr_start() { counter_reset(); for (int i = 0; i < 8; i++) counter_configure(i, ctr_ids[i]); }
static void ctr_stop() { counter_snapshot_take(); for (int i = 0; i < 8; i++) ctr_vals[i] = counter_read(i); counter_snapshot_reset(); }
static void report(const char * name, uint64_t start, uint64_t issued, uint64_t end, int pairs) {
  printf("%-14s total %6llu  cpu_issue_done %6llu  pairs %3d  cyc/pair %4llu |",
    name, (unsigned long long)(end - start), (unsigned long long)(issued - start), pairs,
    (unsigned long long)((end - start) / pairs));
  for (int i = 0; i < 8; i++) printf(" %s=%u", ctr_names[i], ctr_vals[i]);
  printf("\n");
}
static int check(const char * name) {
  gemmini_config_st(NM_J * sizeof(acc_t));
  for (int it = 0; it < IT; it++)
    for (int jt = 0; jt < JT; jt++)
      gemmini_mvout(&c_out[it * DIM][jt * DIM], ACC_BIT | FULL_BIT | C_ACC(it, jt));
  gemmini_fence();
  for (int y = 0; y < NM_I; y++)
    for (int x = 0; x < NM_J; x++)
      if (c_out[y][x] != gold[y][x]) {
        printf("%s mismatch at (%d,%d): got %d want %d\n", name, y, x, c_out[y][x], gold[y][x]);
        return 1;
      }
  printf("%s matches the cpu reference (%d values)\n", name, NM_I * NM_J);
  return 0;
}

// ---- dense: same code as berkeley_dense_perf.c ----
static void dense_fresh(const char * name) {
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_fence();
  ctr_start();
  uint64_t start = read_cycles();
  for (int it = 0; it < IT; it++)
    for (int jt = 0; jt < JT; jt++)
      for (int kt = 0; kt < KT; kt++) {
        uint32_t c = ACC_BIT | (kt > 0 ? ACCUM_BIT : 0) | C_ACC(it, jt);
        PRELOAD(W_SP(kt, jt), c);
        COMPUTE_PRE(A_SP(it, kt), GARBAGE_ADDR);
      }
  uint64_t issued = read_cycles();
  gemmini_fence();
  uint64_t end = read_cycles();
  ctr_stop();
  report(name, start, issued, end, IT * JT * KT);
}
static void dense_reuse(const char * name) {
  const int blocks = JT * KT;
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_fence();
  ctr_start();
  uint64_t start = read_cycles();
  for (int b = 0; b < blocks; b++) {
    const int jt = b / KT, kt = b % KT;
    for (int it = 0; it < IT; it++) {
      uint32_t c = ACC_BIT | (kt > 0 ? ACCUM_BIT : 0) | C_ACC(it, jt);
      if (it == 0) { PRELOAD(W_SP(kt, jt), c); }
      else { PRELOAD(GARBAGE_ADDR, c); }
      if (it == 0) { COMPUTE_PRE(A_SP(it, kt), GARBAGE_ADDR); }
      else { COMPUTE_ACC(A_SP(it, kt), GARBAGE_ADDR); }
    }
  }
  uint64_t issued = read_cycles();
  gemmini_fence();
  uint64_t end = read_cycles();
  ctr_stop();
  report(name, start, issued, end, blocks * IT);
}

// ---- sparse: the same two loops, with the squashed weights and their position notes ----
static void sparse_fresh(const char * name) {
  const int ntiles = IT * JT * KCT;
  gemmini_nm_config_ex(1, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 1, 1);
  gemmini_fence();
  ctr_start();
  uint64_t start = read_cycles();
  send_meta(meta[0][0]);
  for (int it = 0; it < IT; it++)
    for (int jt = 0; jt < JT; jt++)
      for (int t = 0; t < KCT; t++) {
        uint32_t c = ACC_BIT | (t > 0 ? ACCUM_BIT : 0) | C_ACC(it, jt);
        PRELOAD(WC_SP(t, jt), c);
        // next tile's notes go between this preload and its compute, so the hardware can overlap them
        if (t + 1 < KCT) send_meta(meta[t + 1][jt]);
        else if (jt + 1 < JT) send_meta(meta[0][jt + 1]);
        else if (it + 1 < IT) send_meta(meta[0][0]);
        COMPUTE_PRE(A_SP(it, 2 * t), A_SP(it, 2 * t + 1));
      }
  uint64_t issued = read_cycles();
  gemmini_fence();
  uint64_t end = read_cycles();
  ctr_stop();
  report(name, start, issued, end, ntiles);
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_fence();
}
static void sparse_reuse(const char * name) {
  const int blocks = JT * KCT;
  gemmini_nm_config_ex(1, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 1, 1);
  gemmini_fence();
  ctr_start();
  uint64_t start = read_cycles();
  send_meta(meta[0][0]);
  for (int b = 0; b < blocks; b++) {
    const int jt = b / KCT, t = b % KCT;
    for (int it = 0; it < IT; it++) {
      uint32_t c = ACC_BIT | (t > 0 ? ACCUM_BIT : 0) | C_ACC(it, jt);
      if (it == 0) { PRELOAD(WC_SP(t, jt), c); }
      else { PRELOAD(GARBAGE_ADDR, c); }
      if (it == IT - 1 && b + 1 < blocks) send_meta(meta[(b + 1) % KCT][(b + 1) / KCT]);
      if (it == 0) { COMPUTE_PRE(A_SP(it, 2 * t), A_SP(it, 2 * t + 1)); }
      else { COMPUTE_ACC(A_SP(it, 2 * t), A_SP(it, 2 * t + 1)); }
    }
  }
  uint64_t issued = read_cycles();
  gemmini_fence();
  uint64_t end = read_cycles();
  ctr_stop();
  report(name, start, issued, end, blocks * IT);
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_fence();
}

int main() {
  printf("fair_perf: %d:%d, i=%d k=%d j=%d\n", NM_N, NM_M, NM_I, NM_K, NM_J);
  gemmini_flush(0);
  for (int y = 0; y < NM_I; y++)
    for (int z = 0; z < NM_K; z++)
      a[y][z] = (rand() % 16) - 8;
  for (int z = 0; z < NM_K; z++)
    for (int x = 0; x < NM_J; x++)
      w[z][x] = (rand() % 16) - 8;
  nm_prune(NM_K, NM_J, &w[0][0], NM_J, NM_N, NM_M);
  nm_compress(NM_K, NM_J, &w[0][0], NM_J, &wc[0][0], &idx[0][0], NM_J, NM_N, NM_M);
  nm_ref_dense(NM_I, NM_K, NM_J, &a[0][0], NM_K, &w[0][0], NM_J, &gold[0][0], NM_J);
  for (int t = 0; t < KCT; t++)
    for (int jt = 0; jt < JT; jt++)
      nm_pack_meta(&idx[t * DIM][jt * DIM], NM_J, NM_M, meta[t][jt]);

  gemmini_config_ld(NM_K * sizeof(elem_t));
  for (int it = 0; it < IT; it++)
    for (int kt = 0; kt < KT; kt++)
      gemmini_mvin(&a[it * DIM][kt * DIM], A_SP(it, kt));
  gemmini_config_ld(NM_J * sizeof(elem_t));
  for (int kt = 0; kt < KT; kt++)
    for (int jt = 0; jt < JT; jt++)
      gemmini_mvin(&w[kt * DIM][jt * DIM], W_SP(kt, jt));
  for (int t = 0; t < KCT; t++)
    for (int jt = 0; jt < JT; jt++)
      gemmini_mvin(&wc[t * DIM][jt * DIM], WC_SP(t, jt));
  gemmini_fence();

  int fails = 0;
  dense_fresh("dense");          fails += check("dense");
  dense_reuse("dense_reuse");    fails += check("dense_reuse");
#ifndef DENSE_ONLY
  sparse_fresh("sparse");        fails += check("sparse");
  sparse_reuse("sparse_reuse");  fails += check("sparse_reuse");
#endif

  printf("cpu only (no gemmini instructions issued):\n");
  cpu_only = 1;
  dense_fresh("cpu_dense");
  dense_reuse("cpu_dense_reuse");
#ifndef DENSE_ONLY
  sparse_fresh("cpu_sparse");
  sparse_reuse("cpu_sparse_reuse");
#endif
  cpu_only = 0;

  if (fails) { printf("fair_perf FAILED\n"); exit(1); }
  printf("fair_perf passed\n");
  exit(0);
}
