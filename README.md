# RAG-master2

Reproducible migration of the financial RAG analysis workflow onto a repo-local layout.

## Layout

- `finance-files/`: private PDF source corpus for retrieval; the repo keeps only a placeholder
- `data/in/`: private CSV transcript inputs; the repo keeps only a placeholder
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
- Acquire the transcript CSV input files for `data/in/` from Dr. Allen Hu.
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

`odean_video_analysis.py` reads:

- `data/in/content_odean.csv`
- `data/in/word_panel_odean.csv`

It takes the 49 `id` values from `content_odean.csv`, reconstructs each transcript by concatenating the word-level `Text` rows from `word_panel_odean.csv`, and runs the shared RAG pipeline on each video.

`odean_matches_analysis.py` reads:

- `data/in/best_advice_match_per_odean.csv`
- `data/in/worst_advice_match_per_odean.csv`
- `data/in/google_sentence_panel_batch1_20260128.csv`

It reconstructs transcripts from `b1_id` / `VideoID` and runs the same RAG pipeline used by `odean_video_analysis.py` on each of the 49 matched videos in each set.

## Run Prompt 6 Segment Batch Analysis

The segment batch runner applies the same `RAG_PROMPT_6` pipeline used by `scripts/prompt6_batch_analysis.py`, but runs one OpenAI Batch request per `(VideoID, segment_no)` row from:

```text
data/in/speakerseg_batch1_20260121.csv
```

Segments with fewer than 10 words are not sent to OpenAI. They are written directly to the output CSV with `video_ID`, `segment_no`, and `text_length` preserved and all analysis/RAG fields left blank.

Default output and state paths:

```text
data/out/prompt-6/speakerseg_batch1_20260121_segment_analysis_results.csv
data/out/prompt-6/speakerseg_batch1_20260121_segment_batch_state/
```

Check local status without calling the OpenAI API:

```bash
./.venv/bin/python scripts/prompt6_segment_batch_analysis.py status
```

Submit missing segments as OpenAI Batch jobs:

```bash
./.venv/bin/python scripts/prompt6_segment_batch_analysis.py submit
```

Poll jobs once and merge completed outputs:

```bash
./.venv/bin/python scripts/prompt6_segment_batch_analysis.py poll
```

Keep polling until all tracked jobs are terminal:

```bash
./.venv/bin/python scripts/prompt6_segment_batch_analysis.py poll --watch
```

Submit missing segments and then poll in one command:

```bash
./.venv/bin/python scripts/prompt6_segment_batch_analysis.py run --watch
```

Useful options:

```bash
./.venv/bin/python scripts/prompt6_segment_batch_analysis.py submit --limit 1000
./.venv/bin/python scripts/prompt6_segment_batch_analysis.py submit --embedding-batch-size 256
./.venv/bin/python scripts/prompt6_segment_batch_analysis.py submit --rebuild-index
./.venv/bin/python scripts/prompt6_segment_batch_analysis.py submit --overwrite
./.venv/bin/python scripts/prompt6_segment_batch_analysis.py submit --retry-errors
```

The runner defaults to 1,000 requests per OpenAI Batch job and 256 segment queries per synchronous embeddings HTTP request while preparing RAG context. It maintains local manifests, request JSONL files, downloaded output/error JSONL files, and merge status in the state directory, so rerunning `submit`, `poll`, or `run --watch` is intended to resume safely from existing state.

If Batch API line items fail after polling, retry only those failed segment requests with:

```bash
./.venv/bin/python scripts/prompt6_segment_batch_analysis.py submit --retry-errors
./.venv/bin/python scripts/prompt6_segment_batch_analysis.py poll --watch
```

`--retry-errors` scans downloaded `batch-*.error.jsonl` files, excludes segments that already have a later successful output or are already in an active batch, and creates new retry batches at the next local batch number. Historical error files are kept for audit; successful retry merges overwrite the existing fallback rows in the final CSV.

## Notes

- `finance-files/` and `data/in/` are intentionally left empty in git except for placeholder files; request the private materials from Dr. Allen Hu.
- `embeddings_index/` is treated as a derived artifact and is intentionally ignored by git.
- If no local FAISS index exists, the script will build one from `finance-files/` before analyzing transcripts.
- The script resumes from the existing CSV by `video_ID` and `segment_no` unless `--overwrite` is provided.
