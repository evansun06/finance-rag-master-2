# RAG-master2

Reproducible migration of the financial RAG analysis workflow onto a repo-local layout.

## Layout

- `finance-files/`: private PDF source corpus for retrieval; the repo keeps only a placeholder
- `data/in/`: private transcript `.txt` inputs; the repo keeps only a placeholder
- `data/out/odean_analysis_results.csv`: generated analysis output
- `embeddings_index/`: generated FAISS index, rebuilt locally as needed
- `utils/`: reusable helpers for embeddings, CSV handling, and JSON serialization
- `scripts/odean_video_analysis.py`: the single executable analysis script

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

By default the script reads transcript files from:

```text
data/in/
```

The analysis script expects local transcript `.txt` files in that directory. Override any path defaults in `.env` if needed. Shared repo paths live in `config.py`.
Model choice, embedding model, retrieval `K`, and related runtime settings live as explicit constants in `scripts/Odean_video_analysis.py` so each script can own its own RAG behavior.

## Run

```bash
python scripts/Odean_video_analysis.py
```

Optional flags:

```bash
python scripts/Odean_video_analysis.py --rebuild-index
python scripts/Odean_video_analysis.py --overwrite
```

## Notes

- `finance-files/` and `data/in/` are intentionally left empty in git except for placeholder files; request the private materials from Dr. Allen Hu.
- `embeddings_index/` is treated as a derived artifact and is intentionally ignored by git.
- If no local FAISS index exists, the script will build one from `finance-files/` before analyzing transcripts.
- The script resumes from the existing CSV by `video_ID` unless `--overwrite` is provided.
