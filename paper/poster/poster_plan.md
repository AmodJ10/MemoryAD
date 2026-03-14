# MemoryAD Poster Plan

## 1. Poster narrative
MemoryAD solves continual anomaly detection for industrial inspection when new product categories arrive over time. The poster story is: existing trainable CIL methods forget old categories, while MemoryAD avoids weight updates and therefore minimizes forgetting by design. The method uses a simple three-part pipeline (Frozen DINOv2 features -> Adaptive Coreset Manager -> k-NN scorer) so new categories can be added without retraining from scratch. Results emphasize stable incremental performance, strong final accuracy, and practical deployment metrics.

## 2. Section-by-section content mapping

### Header / Title box
- Poster title: MemoryAD: Training-Free Continual Anomaly Detection with Adaptive Coreset Memory
- Subtitle line: Architecture-Agnostic Continual AD for Industrial Inspection
- Admin placeholders: Project ID: [TO BE FILLED], Guide: [TO BE FILLED], Team: [TO BE FILLED]
- Canonical source:
  - paper/main.tex (title, abstract framing)
  - README.md (project summary)

### Introduction box
- Draft text:
  - Industrial anomaly detection learns normal visual patterns to detect defects. In practice, product categories appear sequentially, so systems must adapt to new categories without retraining on all historical data. This creates catastrophic forgetting in many continual-learning pipelines. MemoryAD addresses this with a training-free design that keeps old knowledge in memory instead of updating model weights.
- Canonical source:
  - paper/main.tex (Abstract + Introduction)

### Literature survey box
- Draft bullets:
  - Conventional AD (PatchCore, PaDiM, EfficientAD, RD4AD) is strong for static single-stage settings but does not natively handle category-incremental updates.
  - Continual learning methods (EWC, LwF, replay) assume trainable parameters and classification-style objectives.
  - For anomaly detection, this mismatch leads to complexity and forgetting; MemoryAD instead freezes features and manages memory directly.
- Canonical source:
  - paper/main.tex (Related Work)

### Problem definition and objectives box
- Problem statement:
  - Given tasks T1..TT, each task introduces new normal-only categories. After each task, the system must detect anomalies across all seen categories under fixed memory budget B.
- Project objectives:
  - Adapt to new product categories incrementally.
  - Avoid retraining from scratch and avoid weight updates.
  - Maintain bounded memory with global budget B=10K.
  - Minimize forgetting across earlier categories.
  - Evaluate on MVTec AD and VisA with incremental metrics.
- Canonical source:
  - paper/main.tex (Problem formulation)
  - README.md (project objective and setup)

### Model architecture box
- Poster wording:
  - Stage 1: Frozen DINOv2 ViT-B/14 extracts patch descriptors (layers [7,11], dim 1536).
  - Stage 2: Adaptive coreset manager allocates per-category memory under global budget and truncates old coresets by greedy prefix.
  - Stage 3: k-NN scoring (k=9, L2) computes patch anomaly scores and max image score.
  - Key claim: no gradient updates -> near-zero forgetting by design; only memory compression can affect old tasks.
- Canonical source:
  - paper/figures/f1_architecture.tex
  - src/pipeline.py
  - src/coreset/adaptive_manager.py
  - configs/default.yaml

### Results and analysis box
- Main claims to show:
  - MVTec AD (5-task): 94.0 +/- 0.8 I-AUROC, FR=0.97, P-AUROC=96.0.
  - VisA (4-task): 90.3 I-AUROC, FR=0.52.
  - Throughput: 29.6 ms/image; memory at B=10K: 117.2 MB.
  - Incremental trend remains stable across tasks.
- Canonical source:
  - paper/tables/t1_main_results.tex
  - paper/tables/t2_visa_results.tex
  - paper/tables/t3_inference_speed.tex
  - paper/tables/t6_memory_footprint.tex
  - paper/figures/f2_incremental_auroc.pdf
  - paper/figures/f7_qualitative_heatmaps.pdf

### Conclusion and future scope box
- Conclusion bullets:
  - MemoryAD delivers high continual AD performance with near-zero forgetting by design via frozen features + adaptive memory + k-NN scoring.
  - The method is practical for production updates because new categories are added without retraining the model.
- Future scope bullets:
  - Improve fine-grained localization with multi-scale features.
  - Extend to stronger domain-shift settings.
  - Optimize nearest-neighbor search for higher throughput.
- Canonical source:
  - paper/main.tex (Conclusion + Limitations + Future work)

## 3. Asset reuse plan
Use exactly these existing assets:
1. Motivation visual: paper/figures/f1_teaser.pdf
2. Main trend figure: paper/figures/f2_incremental_auroc.pdf
3. Qualitative evidence: paper/figures/f7_qualitative_heatmaps.pdf
4. Compact comparison table (manual compact version from): paper/tables/t1_main_results.tex
5. Optional compact callout table source (one small metric strip): paper/tables/t3_inference_speed.tex (29.6 ms/image) and paper/tables/t6_memory_footprint.tex (117.2 MB at B=10K)

Note on architecture visual:
- Use a simplified poster diagram that follows the same 3-stage flow as paper/figures/f1_architecture.tex, but with larger text for 10-second readability.

## 4. Layout implementation plan
- Keep structure close to the template:
  - Top full-width header block.
  - Upper row: Introduction | Literature Survey | Problem Definition & Objectives.
  - Middle row: Model Architecture (wide) + Results and Analysis (wide).
  - Bottom row: Conclusion and Future Scope.
- Poster design choices for readability:
  - Large title and section headers.
  - Short bullets and short claim statements.
  - One compact numeric table + one trend plot + one qualitative panel.
  - Metric chips for key deployment values (B=10K, k=9, 29.6 ms/image, 117.2 MB).
- LaTeX strategy:
  - Single file poster scaffold in paper/poster/poster.tex.
  - Use article + geometry + tikz + tcolorbox for predictable compile and rigid box layout.
  - Reference assets using relative paths ../figures/.
  - Keep placeholders for non-research fields (Project ID, Department code, Session, etc.).

## 5. Verification plan
1. Compile check:
- Build paper/poster/poster.tex and ensure no missing asset paths.

2. Metric consistency check:
- MVTec main metrics from paper/tables/t1_main_results.tex.
- VisA metric from paper/tables/t2_visa_results.tex.
- Throughput from paper/tables/t3_inference_speed.tex.
- Memory from paper/tables/t6_memory_footprint.tex.

3. Architecture consistency check:
- Confirm poster flow and labels match src/pipeline.py and src/coreset/adaptive_manager.py (frozen feature extraction, coreset update, k-NN scoring).

4. Template fidelity check:
- Confirm final box arrangement stays close to poster_template.png major layout.

5. Administrative placeholders check:
- Confirm placeholder fields exist for Project ID and other institution-specific non-technical entries.
