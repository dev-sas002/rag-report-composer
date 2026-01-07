# Quick Start Guide

Get the RAG Report Generator running in a few minutes. See `README.md` for what
it does and how it is put together.

## Prerequisites

- Python 3.11 or higher (or Docker)
- ~3GB of disk space — the dependency tree includes PyTorch
- An OpenAI API key is **optional**. Without one the app uses a local embedding
  model and an extractive writer, and everything still works end to end.

## Option A: Docker

```bash
docker compose up --build
```

Open <http://localhost:8350/docs>. The container indexes `data/raw_data/` at
startup, so there is something to query immediately.

```bash
curl -X POST http://localhost:8350/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "How much did revenue grow and which segment drove it?"}'
```

## Option B: Local install

### 1. Install

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure (optional)

```bash
cp .env.example .env
```

Every setting has a working default, so this step is only needed to change one.
To use OpenAI, set `OPENAI_API_KEY` in `.env`.

### 3. Check the setup

```bash
python scripts/verify_setup.py
```

### 4. Index the sample data

```bash
python scripts/ingest_sample_data.py
```

The three sample documents describe a fictional company, Northwind Analytics:
an annual report, a customer-satisfaction summary and a product overview.

### 5. Ask a question

```bash
python main.py --query "How much did revenue grow and which segment drove it?"
```

## Example queries

These are all answerable from the sample corpus, so you can check the report
against the source files in `data/raw_data/`:

```bash
python main.py --query "What happened to gross margin and why?"
python main.py --query "What do customers complain about most?"
python main.py --query "What is the current Net Promoter Score?"
python main.py --query "Which modules does the platform provide?" --output --format markdown
```

## Understanding the output

A report contains:

1. **Executive summary** — the condensed version
2. **Detailed report** — the full answer, with a `[S1]`-style marker on every
   claim naming the source it came from
3. **Sources** — each marker resolved to a document and its retrieval distance
4. **Provenance** — the groundedness score, which markers were used, and any
   the model invented
5. **Cost summary** — tokens and USD for the run (zero on the local backend)

`--output` writes the report to `reports/`; `--format markdown` writes `.md`
instead of `.txt`.

## Common commands

```bash
python main.py --stats                        # chunks currently indexed
python main.py --clear                        # drop the collection
python main.py --ingest-dir ./my/documents    # index your own files
python main.py --ingest-file ./report.pdf     # index one file

python scripts/run_evaluation.py              # score retrieval quality
python scripts/benchmark.py                   # measure chunking, caching, cost
python -m pytest -q                           # run the test suite

./scripts/start_api.sh                        # serve the API on :8350
```

Supported document formats: `.pdf`, `.txt`, `.md`, `.docx`, `.doc`.

`make help` lists the equivalent Make targets.

## Docker helper script

```bash
./docker-run.sh build
./docker-run.sh ingest /app/data/raw_data
./docker-run.sh query "What happened to gross margin and why?"
./docker-run.sh stats
./docker-run.sh shell
./docker-run.sh down
```

## Troubleshooting

**"Vector store is empty"** — run `python scripts/ingest_sample_data.py`.

**`embedding_dimension_mismatch` in the logs** — the index was built with a
different embedding backend (switching between OpenAI and local). Run
`python main.py --clear` and re-ingest.

**"Module not found"** — `pip install -r requirements.txt`.

**`CHUNK_OVERLAP must be smaller than CHUNK_SIZE`** — the two values in `.env`
are inconsistent; settings rejects this at startup rather than failing later
during ingestion.

**OpenAI errors** — check the key in `.env`, that the account has credit, and
that `PROVIDER_MODE` is not forcing `openai` without a key. Unset the key to
fall back to the local providers.

**Docker build problems** — `docker compose build --no-cache`.

Application logs are in `logs/app.log`; per-call costs in
`logs/cost_tracking.json`.

## Next steps

- Put your own documents in `data/raw_data/` and re-run the ingestion.
- Tune `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K_RESULTS` and
  `CONTEXT_RELEVANCE_MARGIN` in `.env`, then run
  `python scripts/run_evaluation.py` to see whether quality moved and
  `python scripts/benchmark.py` to see what it cost.
- Read `README.md` for the architecture and the evaluation methodology.
