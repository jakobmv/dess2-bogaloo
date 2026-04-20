# TASKS.md

## Part 1: SQID paper reproduction

- [x] Bootstrap repo with uv, Makefile, src/, scripts/, gitignore
- [x] Add SQID data/resource loading
- [x] Reconstruct the exact paper reranking subset
- [x] Implement corrected paper-style NDCG evaluation
- [x] Run Random baseline
- [ ] Run ESCI_Baseline
- [x] Run SBERT_text
- [x] Run CLIP_text
- [x] Run CLIP_image
- [x] Run text-image fusion variants
- [x] Save one reproduction summary table
- [ ] Confirm scores are close enough to the paper

## Part 2: DESS extension

- [x] Add 3-layer MLP + DESS reranker
- [ ] Reuse official DESS implementation
- [ ] Train DESS on the same reranking task
- [ ] Evaluate DESS with the same corrected NDCG protocol
- [ ] Save one extension summary table
- [ ] Generate qualitative examples with retrieved product images

## Final cleanup

- [x] README with minimal commands
- [x] Makefile targets working
- [ ] No large artifacts committed
- [x] Reproduction and DESS results clearly separated
