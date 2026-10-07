# Verification report

Run 2026-10-07 13:38 with `verification/verify_all.py`. **ALL CHECKS PASSED** (19 checks, 2 skipped).

## A. Original runs and quoted numbers

- PASS: cross_check.py: 59 checks ok, 0 failed

## B. Independent math check

- PASS: sparse math equals dense math in every random case (2:4 and 4:8): 2/2 patterns
- PASS: every injected hardware fault detected: 8/8 fault types
- PASS: output identical to the stored verification/independent_math_check.out

## C. Pattern comparison

- PASS: pattern_fidelity.py reproduces pattern_fidelity.out exactly
- PASS: bell curve: share kept ranks paired 4:8 < 2:4 < 4:8 < no pattern (79.2% < 86.8% < 89.7% < 92.9%)
- PASS: heavy tail: share kept ranks paired 4:8 < 2:4 < 4:8 < no pattern (86.1% < 92.1% < 94.4% < 96.7%)

## D. Fair benchmark

- PASS: berkeley: passed, 2/2 results match the processor's reference answer (2,048 values each)
- PASS: nm24: passed, 4/4 results match the processor's reference answer (2,048 values each)
- PASS: nm48: passed, 4/4 results match the processor's reference answer (2,048 values each)
- PASS: nm24_sameprog: passed, 2/2 results match the processor's reference answer (2,048 values each)
- PASS: 2:4: half the weight blocks (32 vs 64)
- PASS: 2:4: grid working time about halved (1257 -> 672, 1.87x)
- PASS: 4:8: half the weight blocks (32 vs 64)
- PASS: 4:8: grid working time about halved (1257 -> 672, 1.87x)
- PASS: our chip, sparsity off, berkeley's exact program: reuse 1628 vs berkeley 1622 (0.4% apart)
- PASS: our chip, sparsity off, berkeley's exact program: fresh 1604 vs berkeley 1604 (0.0% apart)
- PASS: paper_data.json (the paper's numbers) matches the raw logs

## E. Paper summary PDF

- SKIP: N-M-Sparsity-Paper-Summary.pdf not found (reports were removed; rebuild with tools/report-builders/build_paper.py)

## F. Code patches in the zip

- SKIP: no Gemmini-NM-Sparsity-*.zip in the project folder (remake the zip to test its patches)

## G. File integrity

- PASS: all 66 recorded files unchanged since the manifest was written
