# SiamGNN for SemEval-2027 Task 3: Semantic Change Across Multiple Time Periods

SiamGNN is our system for SemEval-2027 Task 3. It starts from the state-of-the-art pipeline for lexical semantic change, XL-LEXEME embeddings followed by clustering, and extends it with three neural components:

| Component | Where it is used |
|---|---|
| **Transformer** | XL-LEXEME (XLM-R-large) encodes every usage, with the target word marked `<t>…</t>` |
| **Few-shot Siamese network** | shared projection + Siamese similarity head trained on episodes of one word each; Subtask 2 treats the sense definition as a one-shot prototype |
| **Graph neural network** | usages of a word form a kNN graph; residual message passing refines usage states; the learned graph is clustered with correlation clustering; APPNP propagation for Subtask 2 |

Task page: <https://www.changeiskey.org/event/2027-semeval/> · Codabench: <https://www.codabench.org/competitions/17510/>

## Results on the development data

All numbers come from the unmodified official scorer. No evaluation word was used for training or for picking a threshold. Raw results are in `outputs/results/*.json`.

### Subtask 1 (1a BCubed-F1 / 1b Spearman ρ)

Zero-shot protocol: every threshold and model is fit on the 60 synthetic-English training words only. Trained models are averaged over 3 seeds.

| System | synth-EN 1a | synth-EN 1b | Swedish 1a | Swedish 1b |
|---|---|---|---|---|
| One cluster per word | 0.680 | 0.000 | 0.756 | 0.000 |
| XL-LEXEME + Affinity Propagation (default) | – | – | 0.258 | 0.295 |
| XL-LEXEME + DMoN (unsupervised GNN) | – | – | 0.447 | 0.345 |
| **Existing:** XL-LEXEME + agglomerative (t = 0.4) | 0.687 | 0.238 | **0.815** | **0.523** |
| XL-LEXEME + correlation clustering (t = 0.55) | **0.709** | 0.165 | 0.792 | 0.486 |
| Siamese head only (no GNN) | 0.681 ±0.01 | 0.173 ±0.02 | 0.777 ±0.01 | 0.342 ±0.05 |
| **SiamGNN** + correlation clustering | 0.678 ±0.01 | 0.249 ±0.04 | 0.777 ±0.01 | 0.481 ±0.01 |
| **SiamGNN** + agglomerative (seed ensemble) | 0.694 | **0.292** | 0.794 | 0.414 |

Few-shot Swedish protocol: leave-one-word-out, so 9 labelled Swedish words decide the threshold or adapt the model for the 10th.

| System | Swedish 1a | Swedish 1b |
|---|---|---|
| XL-LEXEME + agglomerative | 0.779 | 0.303 |
| XL-LEXEME + correlation clustering | **0.796** | **0.546** |
| SiamGNN + few-shot adaptation | 0.787 ±0.01 | 0.458 ±0.02 |

### Subtask 2 (per-word macro-F1, Swedish, threshold picked leave-one-word-out)

| System | macro-F1 | mean AUC |
|---|---|---|
| **Existing:** cosine(usage, definition) | 0.571 | 0.688 |
| Sentence Transformer (multilingual-E5) cosine | 0.531 | 0.640 |
| APPNP propagation on the usage graph | 0.575 | 0.695 |
| SiamGNN support probability | 0.579 | 0.698 |
| **One-shot prototype refinement** | **0.588** | 0.704 |
| Few-shot calibrator (logistic regression) | 0.516 | **0.746** |

### Findings

1. **Message passing matters.** Two GNN layers on top of the Siamese head raise ρ from 0.17 to 0.25 on synthetic English and from 0.34 to 0.48 on Swedish, consistently over seeds.
2. **In-domain gain, cross-lingual transfer gap.** On unseen synthetic-English words, SiamGNN gives the best sense dynamics (ρ 0.292 vs 0.238). On historical Swedish (OCR text, another language), the WiC-pretrained XL-LEXEME cosine with a transferred threshold stays best. The learned graph over-merges senses: 2.4 vs 3.5 clusters per word against 5.7 gold senses.
3. **Evaluation protocol matters.** Picking the baseline threshold on the test words inflates ρ from 0.30 (leave-one-word-out) to 0.52.
4. **Subtask 2:** one-shot prototype refinement helps (+1.7 F1, +1.6 AUC). Some target senses are not separable with frozen embeddings (e.g. *kvinna* "womankind" vs. "a woman", AUC 0.33). The next step is to fine-tune the Transformer itself.

### Which configuration to submit

We plan to submit two Subtask 1 runs:

- `--s1-method siamgnn`: this project's method, using `models/siamgnn_final.pt`, trained on all 60 synthetic-English words and all 10 Swedish dev words.
- `--s1-method baseline`: the strongest system on the Swedish dev set.

Subtask 2 uses one-shot prototype refinement (`configs/final.json`). End-to-end check on the dev zip with the official CLI: 1a 0.814, 1b 0.588, Subtask 2 0.599. These are in-sample for the final SiamGNN, which saw these words, so use the tables above for fair estimates.

## Repository layout

```
src/mtp/            library code
  data.py           task I/O and submission writer
  encoders.py       XL-LEXEME re-implementation (+ checkpointed embedding cache)
  clustering.py     agglomerative, Affinity Propagation, K-means, correlation clustering, DMoN, graph smoothing
  siamgnn.py        Siamese-GNN usage-graph model (training / inference / save / load)
  subtask2.py       prototype refinement, APPNP propagation, few-shot support scores
  scorer.py         wrapper around the *unmodified* official scorer
scripts/            experiments, figure generation, end-to-end pipeline
external/semeval2027-task3/   official scorer (git submodule, commit 0e69238)
data/               task and synthetic data (not in the repository; see "Getting the data")
models/             trained SiamGNN checkpoints
outputs/results/    every experiment result as JSON
report/             system description paper (ACL template, LaTeX sources, generated tables, PDF figures)
configs/            final.json (submitted hyperparameters), author.example.json (template for the local author.json)
literature/         verified BibTeX of all references
```

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe "transformers>=4.44,<5" "sentence-transformers>=3.1,<6" huggingface_hub scikit-learn scipy numpy pandas bcubed networkx tqdm sentencepiece protobuf matplotlib
```

XL-LEXEME (`pierluigic/xl-lexeme`, 2.2 GB) downloads automatically on first use. Everything runs on CPU. Encoding is the slow step: about 4–9 sentences/s on a laptop i7, so a GPU (Colab or Kaggle) is recommended for the 130-word test set (40 SV, 30 EN, 15 each IT/ES/NL/RU).

## Getting the data

The repository does not include task data, embedding caches or downloaded models. Clone with the official scorer submodule, then fetch the data:

```bash
git clone --recurse-submodules <repository URL>      # or: git submodule update --init
```

- **Development data:** download `dev.zip` from the Codabench competition 17510 (files / public data) and extract it so that `data/raw/traindev_phase/extracted/SV/usages.jsonl` exists.
- **Synthetic English data:** download the archive from Zenodo, DOI 10.5281/zenodo.21703081, and extract it to `data/raw/synthetic_en/extracted/`. Then run `prepare_synthetic_en.py` and `subsample_synthetic.py` as shown below.
- **Models:** XL-LEXEME downloads automatically. The multilingual-E5 model is used only for one Subtask 2 comparison and goes in `models/hf/multilingual-e5-base` (from `intfloat/multilingual-e5-base`). The trained SiamGNN checkpoints (`models/*.pt`) are included.

## Reproduce the dev results

```bash
.venv/Scripts/python.exe scripts/embed_usages.py data/raw/traindev_phase/extracted   # Swedish dev embeddings
.venv/Scripts/python.exe scripts/prepare_synthetic_en.py                             # synthetic English -> task format
.venv/Scripts/python.exe scripts/subsample_synthetic.py                              # TRAIN_SUB / DEV20 / DEV20_SUB splits
.venv/Scripts/python.exe scripts/embed_synthetic.py                                  # TRAIN_SUB + DEV20_SUB embeddings
.venv/Scripts/python.exe scripts/baselines.py                                        # trivial baselines
.venv/Scripts/python.exe scripts/exp_main.py --seeds 0 1 2                           # Subtask 1 main table
.venv/Scripts/python.exe scripts/exp_hybrid.py                                       # similarity x algorithm (appendix)
.venv/Scripts/python.exe scripts/exp_ensemble.py                                     # seed ensembles (appendix)
.venv/Scripts/python.exe scripts/exp_graph_smoothing.py                              # graph feature smoothing (appendix)
.venv/Scripts/python.exe scripts/exp_s2_final.py                                     # Subtask 2 main table
.venv/Scripts/python.exe scripts/report_stats.py                                     # data statistics + per-word analysis
.venv/Scripts/python.exe scripts/make_paper_figures.py                               # paper figures
.venv/Scripts/python.exe scripts/test_cc_solver.py                                   # correlation clustering vs. brute force
```

`subsample_synthetic.py` reproduces the experiment splits byte for byte (15 usages per word and period, fixed seeds).

## Make a Codabench submission (evaluation phase, January 2027)

```bash
# Phase 1 (Subtask 1, 10-21 Jan 2027): unzip the official test data into data/eval1/<LANG>/usages.jsonl
.venv/Scripts/python.exe scripts/run_pipeline.py --data data/eval1 --out outputs/submissions/eval1 --subtasks 1
# Phase 2 (Subtask 2, 21-31 Jan 2027): data/eval2/<LANG>/{usages,definitions}.jsonl
.venv/Scripts/python.exe scripts/run_pipeline.py --data data/eval2 --out outputs/submissions/eval2 --subtasks 2
```

Each run writes `<LANG>_subtask{1,2}.jsonl` files and a zip that is uploaded as is. On the dev data the script also runs the official scorer on the zip as an end-to-end check.

## Author details

The paper's author block is generated from `configs/author.json`, which is git-ignored so that personal details stay local. Copy `configs/author.example.json` to `configs/author.json` and fill it in; without it, the build uses the placeholders. `author_note` adds a footnote to the author name, `acknowledgment_note` a sentence before the acknowledgments, and `code_url` a code footnote.

## System description paper

`scripts/build_report.py` builds `report/paper.pdf`, the system description paper in the official SemEval format. SemEval system papers use the ACL LaTeX style (`report/acl.sty` and `acl_natbib.bst` from github.com/acl-org/acl-style-files, commit d5adc82) and follow the structure of the SemEval system-paper template: Introduction, Background, System Overview, Experimental Setup, Results, Conclusion, then Limitations, Ethical Considerations, references and appendices. The title follows the required form "Team at SemEval-2027 Task 3: Title". The main text fits the 5-page limit for single-task system papers; references and appendices do not count towards it. Papers are not anonymous, so it uses the `[final]` option.

```bash
.venv/Scripts/python.exe scripts/build_report.py      # author block + tables from outputs/results/*.json, then pdflatex/bibtex
```

The `report/` folder is self-contained and can be uploaded to Overleaf as is. The task-paper citation (`tahmasebi2027semeval` in `report/references.bib`) is the placeholder SemEval asks for until the official BibTeX is released at camera-ready. Test-phase results and the ranking still need to be added after January 2027.

## Key references

Cassotti et al. 2023 (XL-LEXEME) · Periti & Tahmasebi 2024 (multi-period LSC) · Schlechtweg et al. 2021 (DWUG, correlation clustering) · Fedorova et al. 2024 (AXOLOTL'24) · Kipf & Welling 2017 (GCN) · Gasteiger et al. 2019 (APPNP) · Tsitsulin et al. 2023 (DMoN) · Snell et al. 2017 (prototypical networks) · Koch et al. 2015 (Siamese one-shot) · Amigó et al. 2009 (BCubed) · Saponara et al. 2026 (synthetic multi-period data). Full BibTeX: `literature/references.bib`.
