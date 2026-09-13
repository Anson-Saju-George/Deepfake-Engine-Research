# Model Inventory — paste into Claude Code at repo root

## ROLE
Read-only inventory pass. Produce `train/MODEL_INVENTORY.md` listing every trained run
actually on disk, with its metrics and training attributes. **Do NOT move, delete,
archive, or modify anything this pass.** Inventory first; cleanup is a separate approved
step.

## DISCOVER (from disk, not from memory or prior docs)
Walk `train/` (and `models/` if still present) and find every completed run — one row per
run. A run = a checkpoint dir containing weights + its summary/metrics files. For each,
pull ground truth from that run's own files (`split_summary.json`, `test_predictions.csv`,
`final_summary.json`, config/args, whatever exists) — never from README claims or
REVISION_AUDIT.md, which may be stale.

## COLUMNS (one row per run)
```
run_id | category (image/spatial/temporal/spatiotemporal) | backbone | head |
dataset_scope | loss | seq_len | stride | seed | params(M) |
test_AUC | test_F1(state which class is positive) | test_acc |
n_test (real/fake) | checkpoint_path | ckpt_size | source_files_found | [FLAGS]
```
- If a value isn't recoverable from that run's files, write `[UNKNOWN]` — do not infer.
- `[FLAGS]`: mark duplicates (same arch/head/scope), runs missing metrics files,
  runs whose test set was 0% real (the degenerate-split bug), and any run_id present on
  disk but absent from the historical 22-run audit (there may be ~28 now — surface the
  extras explicitly).

## TOP OF THE DOC (before the table)
- Total run count found, and how it reconciles against the audit's 22 (list the extras).
- One-line caveat: these metrics were produced under the pre-rebuild protocol
  (combined-corpus training, pre-decode-fix) and are NOT comparable to post-rebuild
  results — retained for provenance only.
- Total disk footprint, and a duplicates/orphans summary.

## RULES
- Read-only. No file moved or changed. `[UNKNOWN]` over guessing.
- Cite the source file each metric came from.
- End by asking me to mark KEEP / ARCHIVE per row — do not decide yourself.

After I mark the rows, the NEXT (separate) step will be: move ARCHIVE runs to
`train/archive/` with a manifest, move junk to `train/temp_trash/`, verify completeness
by checksum+count BEFORE anything leaves the active tree, and leave `train/` clean and
ready for the rebuild. Do not start that until I approve the marked list.
