// dense-only copy of nm_sparse_perf.c, for the UNMODIFIED berkeley gemmini (GemminiRocketConfig).
// same matrices (same rand() sequence, same 2:4 pruning), same scratchpad layout, same timing loops
// and counters as the dense runs in nm_sparse_perf, so the numbers line up row for row.
// stock gemmini has no sparse mode, so the sparse runs are left out. adds a full cpu reference
// check of the dense result. nothing here touches the gemmini repos; built into verification/bin.

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/gemmini_counter.h"
#include "include/gemmini_nm.h"   // only for nm_prune / nm_ref_dense (plain c helpers)

#define NM_I (4*DIM)
#define NM_K (8*DIM)
#define NM_J (2*DIM)
#define IT (NM_I / DIM)
#define KT (NM_K / DIM)
#define JT (NM_J / DIM)

static elem_t a[NM_I][NM_K] row_align(1);
static elem_t w[NM_K][NM_J] row_align(1);
static acc_t c_dense[NM_I][NM_J] row_align_acc(1);
static acc_t gold[NM_I][NM_J];

#define A_SP(it, kt) (((kt) % 2) * BANK_ROWS + ((it) * (KT / 2) + (kt) / 2) * DIM)
#define W_SP(kt, jt) (2 * BANK_ROWS + ((kt) * JT + (jt)) * DIM)
#define ACC_BIT ((uint32_t)1 << (ADDR_LEN - 1))
#define ACCUM_BIT ((uint32_t)1 << (ADDR_LEN - 2))
#define FULL_BIT ((uint32_t)1 << (ADDR_LEN - 3))
#define C_ACC(it, jt) (((it) * JT + (jt)) * DIM)

static int cpu_only = 0;
#define SINK(x, y) asm volatile ("" :: "r"((uint64_t)(x)), "r"((uint64_t)(y)))
#define PRELOAD(b, c) do { if (cpu_only) SINK(b, c); else { gemmini_preload(b, c); } } while (0)
#define COMPUTE_PRE(a, bd) do { if (cpu_only) SINK(a, bd); else { gemmini_compute_preloaded(a, bd); } } while (0)
#define COMPUTE_ACC(a, bd) do { if (cpu_only) SINK(a, bd); else { gemmini_compute_accumulated(a, bd); } } while (0)

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

static void store_c(void) {
  gemmini_config_st(NM_J * sizeof(acc_t));
  for (int it = 0; it < IT; it++)
    for (int jt = 0; jt < JT; jt++)
      gemmini_mvout(&c_dense[it * DIM][jt * DIM], ACC_BIT | FULL_BIT | C_ACC(it, jt));
  gemmini_fence();
}

static int check(const char * name) {
  for (int y = 0; y < NM_I; y++)
    for (int x = 0; x < NM_J; x++)
      if (c_dense[y][x] != gold[y][x]) {
        printf("%s mismatch at (%d,%d): got %d want %d\n", name, y, x, c_dense[y][x], gold[y][x]);
        return 1;
      }
  printf("%s matches the cpu reference (%d values)\n", name, NM_I * NM_J);
  return 0;
}

static void run_dense(int bias, const char * name) {
  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);
  gemmini_fence();
  ctr_start();
  uint64_t start = read_cycles();
  for (int it = 0; it < IT; it++)
    for (int jt = 0; jt < JT; jt++)
      for (int kt = 0; kt < KT; kt++) {
        uint32_t c = ACC_BIT | (kt > 0 ? ACCUM_BIT : 0) | C_ACC(it, jt);
        PRELOAD(W_SP(kt, jt), c);
        COMPUTE_PRE(A_SP(it, kt), bias ? A_SP(it, kt ^ 1) : GARBAGE_ADDR);
      }
  uint64_t issued = read_cycles();
  gemmini_fence();
  uint64_t end = read_cycles();
  ctr_stop();
  report(name, start, issued, end, IT * JT * KT);
}

static void run_reuse(const char * name) {
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

int main() {
  printf("berkeley_dense_perf: unmodified gemmini, i=%d k=%d j=%d\n", NM_I, NM_K, NM_J);
  gemmini_flush(0);

  // identical data to nm_sparse_perf (same rand() calls in the same order, same 2:4 pruning)
  for (int y = 0; y < NM_I; y++)
    for (int z = 0; z < NM_K; z++)
      a[y][z] = (rand() % 16) - 8;
  for (int z = 0; z < NM_K; z++)
    for (int x = 0; x < NM_J; x++)
      w[z][x] = (rand() % 16) - 8;
  nm_prune(NM_K, NM_J, &w[0][0], NM_J, 2, 4);
  nm_ref_dense(NM_I, NM_K, NM_J, &a[0][0], NM_K, &w[0][0], NM_J, &gold[0][0], NM_J);

  gemmini_config_ld(NM_K * sizeof(elem_t));
  for (int it = 0; it < IT; it++)
    for (int kt = 0; kt < KT; kt++)
      gemmini_mvin(&a[it * DIM][kt * DIM], A_SP(it, kt));
  gemmini_config_ld(NM_J * sizeof(elem_t));
  for (int kt = 0; kt < KT; kt++)
    for (int jt = 0; jt < JT; jt++)
      gemmini_mvin(&w[kt * DIM][jt * DIM], W_SP(kt, jt));
  gemmini_fence();

  int fails = 0;
  run_dense(0, "dense");
  store_c(); fails += check("dense");
  run_dense(1, "dense_bias");
  run_reuse("dense_reuse");
  store_c(); fails += check("dense_reuse");

  printf("cpu only (no gemmini instructions issued):\n");
  cpu_only = 1;
  run_dense(0, "cpu_dense");
  run_reuse("cpu_dense_reuse");
  cpu_only = 0;

  if (fails) { printf("berkeley_dense_perf FAILED\n"); exit(1); }
  printf("berkeley_dense_perf passed\n");
  exit(0);
}
