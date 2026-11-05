# Reviewer 1 experiment and revision contract

## Scope and source precedence

The starting repository is `5d7492eb13efb81273ea13d53673bba9f1eae80f`. The manuscript SHA256 is `f27175c521d471e65f31d654ae34300ae4e9fab15bcae14af02130bc85aae278`. Raw run artifacts take precedence over aggregate JSON, tables, manuscript prose, and reviewer interpretation, in that order. Conflicting historical artifacts remain visible. New results must not overwrite historical results.

The present evidence is EVIDENCE_THIN: unequal evaluation spans and unequal CPT trainability undermine the primary comparison. The task is authorized experiment implementation and execution on avi-gn-fsk, followed by minimal evidence-supported revisions. Acceptance and the identity of any reviewer model are not measurable endpoints.

## Discovery before code

`aratoken.training.prepare_model` owns freezing and layer selection; `scripts.lep.train` consumes Hydra configuration and a DatasetDict with train/validation keys. Extend these interfaces with explicit embedding training rather than writing a trainer. `aratoken.init.build` already accepts seed and alpha; use it for both components and mixtures after correcting its inconsistent old-vocabulary boundary and dtype. `scripts.lep.make_grid.run` and `COLUMNS` define the existing manifest contract and will be reused. `aratoken.metrics.per_doc_bpc` owns BPC; preserve its historical mode while adding a common canonical text path. The existing `load_causal_lm`, `load_texts`, and `paper_normalizer` supply inference loading and input preparation. No generation benchmark currently exists.

## Predeclared questions

1. Does LEP improve canonical-text BPC against CPT when both permit embedding training and comparable transformer layers? Frozen-old-row LEP is compared with CPT training all original embedding rows, plus an all-parameters pair. Equal token exposure is not equal FLOPs or equal text exposure; measured training time and parameter counts accompany it.
2. Does the half-weight initializer improve on compositional-only and analogical-only initialization at the original budget, at approximately 500M tokens, and at 1.7B? Random and moment-matched initializers provide additional controls. All use the same dtype, vocabulary boundary, seed policy, and optimizer schedule.
3. Is the half-weight result robust over alpha in {0, .25, .5, .75, 1}? This original-budget sensitivity study does not select a new primary alpha; .5 remains primary regardless of outcome.
4. What are inference latency, token throughput, character throughput, and peak memory under identical prompts and decoding settings?

The component table will show every planned endpoint and mixture. A synergy claim requires the mixture to improve on both components with uncertainty that supports that conclusion. Null or adverse results instead establish a boundary; they are not discarded.

## Frozen experiment configuration

- Models: Qwen3-0.6B-Base and Qwen3-1.7B-Base; existing released extended tokenizer IDs retained.
- Training seeds: 1, 2, 3; initializer seeds also 1, 2, 3. Deterministic initializers may produce identical starting weights across seeds, which is recorded.
- Token batch: nominally 96 packed sequences of 1024 tokens per optimizer step. Original budget: 2,000 steps (nominal 196,608,000 packed tokens). Extended budget: 5,087 steps (nominal 500,072,448 packed tokens). Actual input-token counts are recorded because a partial accumulation group can occur at an epoch boundary; these are not the count of unmasked prediction targets.
- Optimizer: existing AdamW, learning rate 2e-4, linear decay, warmup ratio .1, weight decay .01 with frozen-row tables excluded from decay. No tuning based on test results.
- Layer sets: last four layers; all model parameters in the fully trainable pair. New embedding rows train for LEP; all original rows train for fair CPT.
- Checkpoint selection: final completed step, `load_best_model_at_end=false`. Training/development losses are diagnostics, not selection criteria.
- Initializer methods: random, mean, moment-matched mean, analogical character-ngram transfer, and hybrid. Historical WECHSEL-style and FOCUS-style implementations coincide for this vocabulary superset and are reported as proxies, not implementations of the published methods.
- Data: existing disjoint lm-train/dev/test artifacts and released tokenized streams. Hashes of actual artifacts are recorded before execution. Existing test data have already been inspected: these are prospective replication runs on a previously used test set, not a new untouched confirmation set.
- Environment repair: the available packed Arrow datasets contain the `List` feature introduced after datasets 3.6.0. Use datasets 4.0.0, which reads them, and retain the other pinned model-library versions; the full regression suite is rerun in this environment.
- Evaluation: canonical text uses the existing alif-preserving paper normalizer, iterated to its fixed point identically before selecting a 512-Unicode-character prefix. A second pass is necessary for characters such as Turkish dotted capital I: lowercasing introduces a combining dot after the accent-stripping stage. The tokenizer and training behavior remain unchanged. All tokenized target tokens are scored from a BOS/EOS conditioning token without tokenizer-specific truncation. Identical canonical span hashes and denominators are required across every compared run. Overflow or normalization changes fail explicitly. Report macro document BPC and corpus BPC separately. Repeat on the first 1,500 eligible test documents, with an English retention evaluation using the same stated normalization policy.
- Primary estimand: seed-mean difference in macro document BPC, hybrid minus fair CPT, separately for each scale/budget. Report individual seed outcomes and seed SD; document resampling is conditional on those training seeds and must not be described as training-seed uncertainty. Use paired seed/document hierarchical resampling for new inferential summaries, with plus-one Monte Carlo p-values and Holm correction over declared contrasts.
- Initializer contrast family: hybrid versus mean, analogical, random, calibrated at each of three scale/budget settings. Fairness family: hybrid versus CPT for partial and full training at each setting. Alpha sensitivity is exploratory and shown in full.
- Practical threshold: 0.005 BPC is retained as a descriptive reference, not a post-hoc significance criterion. Improvement requires a negative difference; uncertainty crossing zero is unresolved. Practical improvement requires the upper interval bound below -0.005.
- Failures: retain attempted runs and errors. Infrastructure/OOM retries keep the same seed and effective batch; reduce microbatch and increase accumulation only if required and record the change. Nonfinite loss stops the run, with no favorable replacement seed. Resume interrupted runs from their own checkpoints only.
- Stop after the finite declared matrix. No extra benchmark, seed, alpha, or epoch is added because a result is unfavorable.

## Implementation and verification order

1. Regression tests on FSK for CPT embedding updates, tied/untied frozen-row invariance, initializer boundaries and endpoints, and common-text scoring. Observe expected failures, apply minimal fixes, and rerun.
2. Generate a separate manifest using the existing row schema; test unique names, seed coverage, exact budgets, parameter freedom, and all initializer/alpha arms. Keep historical grid unchanged.
3. Record dependency versions, artifact hashes, hardware, and effective command lines. Run short forward/backward/save/reload smoke tests before full runs.
4. Run the complete finite matrix on available FSK GPUs with exclusive advisory leases, append-only event records, logs, and per-run final models. Do not interrupt unrelated jobs.
5. Rescore historical checkpoints with the common-text protocol and benchmark generation; keep corrected and historical metrics clearly separated.
6. Aggregate completed runs, report missing denominators, reconstruct tables, and revise only claims supported by those artifacts.
7. Apply ToughAIPaperReviewer to pinned manuscripts with all eight real domain reviewers, reconcile evidence-backed issues, and rerun affected work. A full protocol audit requires all eight domains; reconnaissance is not represented as that audit.

## Required minimal paper corrections

Explicitly disclose English forgetting and nonsignificant task differences in the abstract; distinguish token compression from throughput; replace named-method proxy comparisons with precise labels; delimit the parallelogram identity as algebra rather than a guarantee of improvement; reconcile 6,602 stored test documents with evaluated subsets; correct p-values and correction families; remove publisher template placeholders; clarify normalization and mixed Arabic/Latin behavior. New numerical claims wait for verified runs.

## Deliverables and unresolved state

Code, tests, frozen matrix, append-only run ledger, complete result tables, reviewer response matrix, clean manuscript, highlighted changes, and a review report are required. Until computation and artifact reconciliation finish, the paper is not submission-ready. AI assistance is recorded in `ai-assistance.jsonl`; author scientific verification and disclosure classification remain required before submission.

## Launch coordination repair, before usable results

The initial multi-host launch exposed host-local `flock` behavior on the shared mount: different nodes claimed identical runs. All campaign processes were stopped and their artifacts quarantined; no affected checkpoint or initializer enters the study. Reuse the existing runner, replacing only cross-host run/initializer claims with atomic directory creation and explicit owner metadata. GPU leases remain host-local. Verify contention from two physical FSK hosts before relaunching the unchanged scientific matrix. This repairs execution ownership, not the scientific protocol. Stale claims require verifying that their recorded process has exited before removing them.
