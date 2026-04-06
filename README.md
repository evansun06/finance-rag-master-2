# RAG-master2

Reproducible migration of the financial RAG analysis workflow onto a repo-local layout.

## Layout

- `finance-files/`: private PDF source corpus for retrieval; the repo keeps only a placeholder
- `data/in/`: private transcript `.txt` inputs; the repo keeps only a placeholder
- `data/out/odean_analysis_results.csv`: generated output for the original per-video Odean analysis
- `data/out/odean_matches_best_analysis_results.csv`: generated output for the 49 best-match videos
- `data/out/odean_matches_worst_analysis_results.csv`: generated output for the 49 worst-match videos
- `embeddings_index/`: generated FAISS index, rebuilt locally as needed
- `utils/`: reusable helpers for embeddings, CSV handling, and JSON serialization
- `scripts/odean_video_analysis.py`: executable script for the original per-video analysis
- `scripts/odean_matches_analysis.py`: executable script for the matched best/worst video analyses

## Setup

Use Python 3.12 and a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Then set `OPENAI_API_KEY` in `.env`.

## Private Data

The private research corpus and raw input files are not stored in this GitHub repo.

- Acquire the `finance-files/` PDF corpus from Dr. Allen Hu.
- Acquire the transcript input files for `data/in/` from Dr. Allen Hu.
- Keep those materials in private cloud storage or on a local synced drive, then place them into the repo-local paths or point the environment variables at their private locations.

## Configure Inputs

By default the data read from scripts/* should be inserted in `data/in/*`:

```text
data/in/
```


## Run
Scripts for different analysis tasks are grouped into `scripts/*`

```bash
python scripts/odean_video_analysis.py
python scripts/odean_matches_analysis.py
```

Optional flags:

```bash
python scripts/odean_video_analysis.py --rebuild-index
python scripts/odean_video_analysis.py --overwrite
python scripts/odean_matches_analysis.py --rebuild-index
python scripts/odean_matches_analysis.py --overwrite
```

`odean_matches_analysis.py` reads:

- `data/in/best_advice_match_per_odean.csv`
- `data/in/worst_advice_match_per_odean.csv`
- `data/in/google_sentence_panel_batch1_20260128.csv`

It reconstructs transcripts from `b1_id` / `VideoID` and runs the same RAG pipeline used by `odean_video_analysis.py` on each of the 49 matched videos in each set.

## Notes

- `finance-files/` and `data/in/` are intentionally left empty in git except for placeholder files; request the private materials from Dr. Allen Hu.
- `embeddings_index/` is treated as a derived artifact and is intentionally ignored by git.
- If no local FAISS index exists, the script will build one from `finance-files/` before analyzing transcripts.
- The script resumes from the existing CSV by `video_ID` unless `--overwrite` is provided.
