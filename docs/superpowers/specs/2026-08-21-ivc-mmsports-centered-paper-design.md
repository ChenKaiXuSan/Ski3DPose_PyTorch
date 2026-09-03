# MMSports-Centered IVC Extension Paper Design

## Objective

Prepare an English regular research article for *Image and Vision Computing*
(IVC) by extending the accepted MMSports CanonFuse3D paper. The MMSports
manuscript remains the conceptual and experimental baseline. The journal paper
adds controlled evidence about synchronization, view geometry, sampling-rate
drift, front-end estimator transfer, reliability-gate behavior, and missing
pose observations without presenting a different architecture as the original
method.

## Immutable source material

- The reference manuscript is
  `paper/mmsports_reference/main_gate_table.tex`.
- The reference bibliography is
  `paper/mmsports_reference/sample-base.bib`.
- Reference figures live under `paper/mmsports_reference/figure/`.
- The reference package is read-only for this project. All IVC files live in a
  separate `paper/ivc_draft_20260821/` directory.
- The MMSports paper title, six-author order, affiliations, ORCID identifiers,
  acknowledgements, core method, datasets, and reported main/ablation results
  are preserved as source facts unless a live code or artifact check proves a
  factual error that must be corrected in the journal version.

## Journal-extension position

The IVC paper is not framed as a new unrelated model. It is a substantial
extension of CanonFuse3D with three additional evidence axes:

1. **Deployment robustness:** temporal offsets, sampling-rate drift, and camera
   view-angle separation.
2. **Model interpretation:** gate/error correlation and masking degradation.
3. **Pipeline generalization:** frozen-model transfer across SAM3D,
   MotionBERT, PoseFormer, and VideoPose3D front ends.

The journal manuscript must cite the MMSports conference version and state the
new material explicitly. Until final proceedings metadata is available, the
conference citation is represented by a visibly marked author-confirmation
entry in the evidence manifest and is not presented as DOI-verified.

## Experimental source of truth

### MMSports baseline

- Unity checkpoint:
  `logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt`
- Unity reference report:
  `logs/experiment3-feature-fusion/full/summary/report.txt`
- MMSports Unity result: canonical-average MPJPE `0.3705`, fused MPJPE
  `0.2711`, and fused acceleration error `0.0312`.
- MMSports Ski-PTZ-Pose result: fused-reference average MPJPE `0.3487`, fused
  MPJPE `0.1794`, and fused acceleration error `0.0311`, as recorded in the
  reference manuscript.

The checkpoint at `2026-05-14/02-46-56` is not the MMSports paper checkpoint.
Results produced from it, including the existing front-end comparison, are
retained as development evidence but are not inserted as final IVC results.

### Journal re-evaluation protocol

All new Unity experiments use the MMSports checkpoint, fold 0, seed 42,
30-frame windows, the same index mapping, and the complete test split. Test
dataloaders set `drop_last=False`; every summary records the exact sample count.
The no-perturbation SAM3D condition is rerun under this complete-split protocol
and anchors every extension table.

The manuscript distinguishes:

- `MMSports-reported`: values reproduced verbatim from the accepted conference
  manuscript and its saved report.
- `IVC complete-split re-evaluation`: values recomputed with the MMSports
  checkpoint after retaining the final partial test batch.

These labels prevent silent mixing of the conference report (`0.2711`) with
development results obtained from a different checkpoint (`0.1751` or
`0.1552`). The paper explains any numerical difference once, in the evaluation
protocol, and uses only complete-split values for comparisons among new
conditions.

### Metrics and units

- Primary spatial metric: MPJPE in the canonical root-relative space.
- Temporal metrics: velocity error and acceleration error in the same space.
- Gate diagnostics: Pearson correlation between alpha and the right-minus-left
  error advantage; preference accuracy; conditional mean alpha.
- Front-end comparison: exact 13-joint intersection, with millimetres used in
  the publication table and explicit conversion from stored metre values.
- Every table states whether lower or higher is better and identifies the
  reference target (Unity recorded avatar skeleton or Ski-PTZ fused canonical
  reference).

## Additional experiments

### Temporal offsets and gate behavior

Perturb the right stream by `-2`, `-1`, `-0.5`, `0`, `0.5`, `1`, and `2`
frames with clamped linear interpolation. Report MPJPE, acceleration error,
absolute and relative degradation from zero offset, gate/error correlation,
and gate preference accuracy.

### Camera-pair view angle

Derive circular azimuth separation from `capture_L*_A*` identifiers. Aggregate
the complete test split into declared 30-degree bins from 0 to 180 degrees.
Report sample count, canonical-average MPJPE, fused MPJPE, fusion gain,
acceleration error, and failure rate for every non-empty bin.

### Sampling-rate drift

Resample the right stream at rate errors `-2%`, `-1%`, `-0.5%`, `0`, `0.5%`,
`1%`, and `2%`, anchored at the sequence center. Report the same spatial and
temporal metrics as the offset study and degradation from zero drift.

### Front-end estimator transfer

Use the existing official VideoPose3D, PoseFormer, and MotionBERT exports, but
evaluate the frozen MMSports checkpoint rather than the development
checkpoint. Every method uses the same Unity H36M-17 ground-truth 2D tracks and
the exact common-13 joint metric. Preserve repository commit hashes,
checkpoint hashes, decoding rules, temporal context, and sample counts in the
manifest. Describe the study as a controlled 2D-to-3D lifting-distribution
transfer experiment, not an end-to-end RGB detector comparison.

### Masking robustness

Rerun or regenerate the zero-to-50% selected masking conditions with the
MMSports checkpoint and complete test split for left-only, right-only, and
both-view corruption under random, distal, and temporal masking. Report raw
observed points and normalized AUC. Do not create error bars because only one
corruption realization exists per condition.

## Manuscript architecture

1. **Title and abstract**: CanonFuse3D remains the named method. The abstract
   defines canonical root-relative 3D pose and leads with the journal extension
   question: how robustly can a frozen calibration-free pose-space fusion model
   transfer beyond its ideal synchronized SAM3D setting?
2. **Introduction**: preserve the practical skiing motivation, state the
   conference origin, and list journal-only contributions separately.
3. **Related work**: monocular lifting, calibrated and uncalibrated multi-view
   pose, unsynchronized sports capture, and pose-estimator transfer.
4. **Method**: retain canonicalization, sequence-level Sim(3) alignment,
   geometric-motion features, bidirectional joint attention, temporal
   refinement, adaptive gate, and residual correction. Correct reproducibility
   details against the active code and checkpoint metadata.
5. **Experimental protocol**: datasets, references, split, checkpoint, missing
   joints, common skeletons, metrics, and complete-split policy.
6. **Core MMSports results**: retain the conference main table and feature
   ablation as the foundation, clearly labeled as reported results.
7. **Robustness and generalization**: temporal/gate, view angle, sampling-rate,
   front-end transfer, and masking subsections.
8. **Discussion**: interpret negative transfer for any front end honestly;
   discuss synchronization, blur, N-view generalization, reference quality,
   single-sport scope, and the distinction between adaptive gates and
   calibrated uncertainty.
9. **Conclusion and declarations**: concise conclusion followed by data/code
   availability, ethics, CRediT, competing interests, funding, and AI-use
   disclosure. Unknown author-controlled declarations are marked as requiring
   author confirmation in the draft package rather than fabricated.

## Deliverable package

`paper/ivc_draft_20260821/` contains:

- `main.tex`: self-contained Elsevier preprint manuscript.
- `references.bib`: only cited records carried from or added to the MMSports
  bibliography, with verification notes kept outside the BibTeX file.
- `tables/*.tex`: one responsibility per result table.
- `figures/`: copied MMSports figures plus generated extension figures.
- `scripts/generate_extension_figures.py`: deterministic figure generation
  from final CSV files.
- `highlights.txt`: candidate journal highlights.
- `cover_letter.md`: conference-extension disclosure and IVC scope fit.
- `evidence_manifest.md`: every manuscript number mapped to a source artifact,
  checkpoint, metric subset, and status.
- `revision_log.md`: section-level account of changes from MMSports.
- compiled `main.pdf` after verification.

## Quality gates

1. The reference manuscript directory has no modifications.
2. Every final extension CSV uses the MMSports checkpoint and reports an exact
   sample count.
3. Zero-condition values agree across temporal, sampling-rate, view-angle,
   masking, and front-end paths within floating-point tolerance when their
   joint subsets match.
4. Every number in the abstract, conclusion, and tables appears in the evidence
   manifest.
5. Every citation key resolves in `references.bib`; new literature is verified
   against a primary source or DOI record.
6. The LaTeX package compiles without undefined references or citations.
7. The rendered PDF is checked for clipped tables, unreadable figures, broken
   equations, and accidental ACM metadata.
8. MMSports-reported and IVC complete-split values are never presented in the
   same comparison column without an explicit protocol label.
