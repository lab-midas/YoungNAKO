# YoungNAKO · Machine Learning (Thu 05.11.2026) · Hands-on material

## Lecturer: [Thomas Küstner](https://www.medizin.uni-tuebingen.de/de/das-klinikum/mitarbeiter/profil/252)

| Notebook | Block | Time | Content |
|---|---|---|---|
| `YN_ML_1_from_regression_to_deep_learning.ipynb`<br>[![Exercise](https://img.shields.io/badge/Exercise-Open_in_Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/lab-midas/YoungNAKO/blob/master/YN_ML_1_from_regression_to_deep_learning.ipynb) [![Solution](https://img.shields.io/badge/Solution-Open_in_Colab-2E7D32?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/lab-midas/YoungNAKO/blob/master/solutions/YN_ML_1_solution.ipynb) | 1 | 30 min | synthetic NAKO-like cohort: logistic regression, random forest, boosting, MLP, TabPFN v2 · first CNN on CHAOS MRI (kidney visible?) |
| `YN_ML_2_beyond_tables.ipynb`<br>[![Exercise](https://img.shields.io/badge/Exercise-Open_in_Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/lab-midas/YoungNAKO/blob/master/YN_ML_2_beyond_tables.ipynb) [![Solution](https://img.shields.io/badge/Solution-Open_in_Colab-2E7D32?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/lab-midas/YoungNAKO/blob/master/solutions/YN_ML_2_solution.ipynb) | 2 | 30 min | TotalSegmentator MRI → Dice, organ volumes, liver fat fraction · DINOv2 embeddings → UMAP, classification, body-part regression · BiomedCLIP zero-shot + mini-CLIP |
| `YN_ML_3_does_it_work.ipynb`<br>[![Exercise](https://img.shields.io/badge/Exercise-Open_in_Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/lab-midas/YoungNAKO/blob/master/YN_ML_3_does_it_work.ipynb) [![Solution](https://img.shields.io/badge/Solution-Open_in_Colab-2E7D32?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/lab-midas/YoungNAKO/blob/master/solutions/YN_ML_3_solution.ipynb) | 3 | 30 min | leakage, domain shift, calibration, Dice vs Hausdorff, shortcut learning (+ occlusion teaser for XAI) |
| `YN_ML_4_at_home_nnunet_clip_nako.ipynb`<br>[![Exercise](https://img.shields.io/badge/Exercise-Open_in_Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/lab-midas/YoungNAKO/blob/master/YN_ML_4_at_home_nnunet_clip_nako.ipynb) | home | – | nnU-Net training on CHAOS (IP+OP), BiomedCLIP fine-tuning, MedSAM2/nnInteractive pointers, pipeline for own NAKO Dixon data (runs locally) |

`solutions/` contains the filled-in versions of notebooks 1–3. Exercise versions have `___` blanks marked **✏️ TODO**.

## Data
CHAOS challenge, MR training set (20 cases, T1-DUAL in-/opposed-phase + liver/kidney/spleen masks), downloaded automatically from Zenodo record 3431873 (≈ 0.9 GB, only the MR part is unpacked). Licence CC BY-NC-SA 4.0. Cite: Kavur et al., *Med Image Anal* 69:101950 (2021). Nothing has to be hosted or shared by us.

## Distribution
Put the notebooks in a public GitHub repo (or Google Drive) and give participants "Open in Colab" links:
`https://colab.research.google.com/github/<user>/<repo>/blob/main/YN_ML_1_from_regression_to_deep_learning.ipynb`

## Local setup (instead of Colab)
Needs Python 3.10–3.12. In a terminal, in this folder:
```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name youngnako      # then pick the "youngnako" kernel in Jupyter / VS Code
```
The setup cell in each notebook (`%pip install ...`) also installs that notebook's packages into whichever kernel is running, so a fresh environment works too; `requirements.txt` just does everything at once.
Locally, data goes to `~/yn_data/chaos` (one download shared by all notebooks, outside the repo/Dropbox) and NB4 works in `~/yn_data/yn_work`. To use other folders, set the environment variables `YN_CHAOS_ROOT` / `YN_WORK`. Without an NVIDIA GPU everything runs on CPU.

## Checklist before the course (please do once on Colab with a T4 GPU)
The notebook logic was executed end-to-end offline on synthetic CHAOS-like data; model downloads (Zenodo, Hugging Face, TotalSegmentator weights, TabPFN weights) could not be reached from the build environment. So:
1. Run all three solution notebooks once on Colab (≈ 10 min each), especially:
   - CHAOS loader: check the overlay figure in NB2 (expert mask aligned with the image). The PNG grey values are mapped 63/126/189/252 → liver / R kidney / L kidney / spleen.
   - NB2 TotalSegmentator `total_mr` (fast mode on CPU, full on GPU) and the kidney left/right convention (Dice ≈ 0 for both kidneys would mean swapped labels).
   - NB1 TabPFN v2 (`ModelVersion.V2`, no PriorLabs login needed).
   - NB3 shortcut part: the "new hospital" AUC should drop clearly; if not, reduce `epochs` of `m_hans` further.
2. Free Colab GPUs are not guaranteed for 40 parallel users: everything also runs on CPU (slower; NB2 uses `fast=True` and fewer cases on CPU).
3. Optional: pre-run the first cell during the break so the 0.9 GB download is cached.

## Own NAKO data
Notebook 4, Part D runs locally (set `DEMO_MODE = False`, `NAKO_ROOT`, `PATTERNS`). Do not upload NAKO data to Colab.
