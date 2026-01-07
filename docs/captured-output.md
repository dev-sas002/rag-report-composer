# Captured output

Everything on this page was produced by running the commands shown, on the
sample corpus in `data/raw_data/`, with **no API key set** — so the embedder is
the local sentence-transformer and the writer is the offline extractive one.
Nothing here is illustrative or hand-edited; the only change is that structured
log lines have been dropped where they would bury the output.

Reproduce it with:

```bash
python scripts/ingest_sample_data.py
python main.py --query "What do customers complain about most?"
python scripts/run_evaluation.py
python scripts/benchmark.py
```

---

## 1. A report from the CLI

```console
$ LOG_LEVEL=WARNING python main.py --query "What do customers complain about most?"
📊 Vector store contains 6 document chunks

🔍 Query: What do customers complain about most?

⏳ Generating report...

================================================================================
EXECUTIVE SUMMARY
================================================================================
## What customers complained about Reporting flexibility remains the largest source of dissatisfaction, named in 44 percent of negative responses. [S1] Customers want to build their own views rather than choose from fixed dashboards. [S1] Export limits were the second complaint, at 27 percent. [S1] Pricing transparency appeared in 19 percent of negative responses, almost entirely from mid-market accounts moving between tiers. [S1]

================================================================================
DETAILED REPORT
================================================================================
## What do customers complain about most

The following passages were retrieved from the indexed documents and are reproduced verbatim.

- ## What customers complained about Reporting flexibility remains the largest source of dissatisfaction, named in 44 percent of negative responses. [S1]

- Customers want to build their own views rather than choose from fixed dashboards. [S1]

- Export limits were the second complaint, at 27 percent. [S1]

- Pricing transparency appeared in 19 percent of negative responses, almost entirely from mid-market accounts moving between tiers. [S1]

- # Customer Satisfaction — Q4 Review ## Headline numbers Net Promoter Score finished the quarter at 41, up from 33 in Q3 and 28 a year ago. [S2]

- Customer Satisfaction (CSAT) on support interactions averaged 4.4 out of 5 across 3,812 rated conversations. [S2]

- Median first-response time fell to 47 minutes from 2 hours 20 minutes, following the move to a follow-the-sun rota across the London and Vancouver teams. [S2]

- ## What customers praised Reliability was the most frequently cited strength, appearing in 61 percent of positive free-text responses. [S2]

---

*Written offline by the extractive writer: every line above appears in the source material. Configure `OPENAI_API_KEY` for a synthesised narrative report.*

================================================================================
SOURCES
================================================================================
[S1] customer_satisfaction_q4.md  (distance 0.896)
[S2] customer_satisfaction_q4.md  (distance 1.207)

================================================================================
PROVENANCE
================================================================================
Groundedness: 1.00  (1.00 = every claim word is in a source)
Cited: S1, S2

================================================================================
COST SUMMARY
================================================================================
Total Cost: $0.0000
Input Tokens: 724
Output Tokens: 375

By Model:
  local-extractive-writer: $0.0000
```

Two things to notice. Retrieval returned five chunks and the report cites two:
the other three were dropped by relevance pruning, because their distance from
the query was more than 35% worse than the best hit. And every quoted line
carries the marker of the source it came from, which is what makes the
groundedness score at the bottom checkable rather than decorative.

---

## 2. The same report over HTTP

```console
$ curl -s http://localhost:8350/health
{
    "status": "healthy",
    "app_name": "RAG Company Report Generator",
    "version": "1.0.0",
    "vector_store_documents": 6
}
```

```console
$ curl -s -X POST http://localhost:8350/query \
    -H 'Content-Type: application/json' \
    -d '{"query": "What do customers complain about most?"}'
{
    "query": "What do customers complain about most?",
    "report": "## What do customers complain about most\n\nThe following passages were retrieved from the indexed documents and are reproduced verbatim.\n\n- ## What customers complained about Reporting flexibility remains the largest source of dissatisfaction, named in 44 percent of negative responses. [S1]\n\n- Customers want to build their own views rather than choose from fixed dashboards. [S1]\n\n- Export limits were the second complaint, at 27 percent. [S1]\n\n- Pricing transparency appeared in 19 percent of negative responses, almost entirely from mid-market accounts moving between tiers. [S1]\n\n- # Customer Satisfaction \u2014 Q4 Review ## Headline numbers Net Promoter Score finished the quarter at 41, up from 33 in Q3 and 28 a year ago. [S2]\n\n- Customer Satisfaction (CSAT) on support interactions averaged 4.4 out of 5 across 3,812 rated conversations. [S2]\n\n- Median first-response time fell to 47 minutes from 2 hours 20 minutes, following the move to a follow-the-sun rota across the London and Vancouver teams. [S2]\n\n- ## What customers praised Reliability was the most frequently cited strength, appearing in 61 percent of positive free-text responses. [S2]\n\n---\n\n*Written offline by the extractive writer: every line above appears in the source material. Configure `OPENAI_API_KEY` for a synthesised narrative report.*",
    "summary": "## What customers complained about Reporting flexibility remains the largest source of dissatisfaction, named in 44 percent of negative responses. [S1] Customers want to build their own views rather than choose from fixed dashboards. [S1] Export limits were the second complaint, at 27 percent. [S1] Pricing transparency appeared in 19 percent of negative responses, almost entirely from mid-market accounts moving between tiers. [S1]",
    "sources": [
        "customer_satisfaction_q4.md",
        "customer_satisfaction_q4.md"
    ],
    "citations": [
        {
            "id": "S1",
            "source": "customer_satisfaction_q4.md",
            "score": 0.8960953787447283,
            "excerpt": "## What customers complained about\n\nReporting flexibility remains the largest source of dissatisfaction, named in 44\npercent of negative responses. Customers want to build their own views rather\nthan choose from fixed dashboards. Export limits were the second complaint, at\n27 per"
        },
        {
            "id": "S2",
            "source": "customer_satisfaction_q4.md",
            "score": 1.2067741668107794,
            "excerpt": "# Customer Satisfaction \u2014 Q4 Review\n\n## Headline numbers\n\nNet Promoter Score finished the quarter at 41, up from 33 in Q3 and 28 a year\nago. Customer Satisfaction (CSAT) on support interactions averaged 4.4 out of 5\nacross 3,812 rated conversations.\n\nMedian first-response time fe"
        }
    ],
    "citation_audit": {
        "valid": [
            "S1",
            "S2"
        ],
        "invalid": [],
        "unused": [],
        "groundedness": 1.0,
        "ungrounded_sample": []
    },
    "groundedness": 1.0,
    "num_tokens_used": 1099,
    "total_cost_usd": 0.0,
    "saved_file": null
}
```

`citations` is the provenance record: which chunk was sent to the model, under
which marker, from which document, at what retrieval distance. `citation_audit`
is the check on what the model did with it.

---

## 3. The audit catching an invented citation

A model asked to cite `[S1]`-style markers will occasionally cite one that was
never supplied. That is a silent failure — the output still looks like a
sourced report. Here the writer is replaced with a stub that does exactly that,
so the audit can be seen firing:

```console
$ python - <<'PY'
from src.agent.graph import ReportGenerationGraph
from src.agent.nodes import AgentNodes
from src.retrieval.vector_store import VectorStore


class OverconfidentWriter:
    """A stand-in for a model that cites a source it was never given."""

    def invoke(self, prompt):
        class R:
            content = (
                "Reporting flexibility is the top complaint [S1], and the "
                "Antarctic division tripled its titanium output [S7]."
            )

        return R()


store = VectorStore()
graph = ReportGenerationGraph(store)
graph.nodes = AgentNodes(store, llm=OverconfidentWriter())
graph.graph = graph._build_graph()

state = graph.generate_report("What do customers complain about most?")
audit = state["citation_audit"]
print("report:      ", state["report"])
print("supplied:    ", [c.id for c in state["citations"]])
print("cited:       ", audit["valid"])
print("INVENTED:    ", audit["invalid"])
print("groundedness:", audit["groundedness"])
print("not in any source:", audit["ungrounded_sample"])
PY

report:       Reporting flexibility is the top complaint [S1], and the Antarctic division tripled its titanium output [S7].
supplied:     ['S1', 'S2']
cited:        ['S1']
INVENTED:     ['S7']
groundedness: 0.3
not in any source: ['antarctic', 'division', 'its', 'output', 'titanium', 'top', 'tripled']
```

`S7` was never supplied, and the words that were invented are named. The graph
also logs `invented_citations` at ERROR when this happens.

---

## 4. The evaluation harness

```console
$ python scripts/run_evaluation.py
backend=local pass=6/6 recall=1.00 precision=0.40 faithfulness=0.98 p95=6399ms

  ✓ revenue-growth     recall=1.00 precision=0.40 faith=0.94 6399ms
  ✓ margin             recall=1.00 precision=0.40 faith=0.98 19ms
  ✓ complaints         recall=1.00 precision=0.40 faith=0.99 6ms
  ✓ nps                recall=1.00 precision=0.40 faith=0.99 18ms
  ✓ modules            recall=1.00 precision=0.40 faith=0.98 33ms
  ✓ deployment         recall=1.00 precision=0.40 faith=0.99 19ms

  citation groundedness: mean 1.00

✅ Evaluation passed.
```

Six questions a human can check by opening the three sample documents. The
script exits non-zero if mean recall drops below 0.80, mean faithfulness below
0.50, or any report cites a source that was never supplied.

Reading these honestly: **recall is the meaningful result** — every question
retrieved the document that answers it, using only the local model.
**Precision of 0.40 is arithmetic, not a failure**: the sample corpus is six
chunks and `TOP_K_RESULTS` is five, so at most two of the five returned chunks
can come from the one expected document. It rises with a larger corpus or a
smaller `k`, and it is reported rather than hidden because a precision number
without its denominator is meaningless.

---

## 5. The scalability benchmark

```console
$ python scripts/benchmark.py
RAG Report Generator — scalability benchmark
Offline: hashing embedder, in-process store, no API key, no network.

1. Chunk size and overlap
-------------------------
Corpus: 3 documents, 3,767 characters

 chunk  overlap  chunks     chars   dup %    embed $
   500        0      10     3,750   -0.5%    0.00012
   500      100      10     3,825    1.5%    0.00012
  1000        0       6     3,758   -0.2%    0.00012
  1000      200       6     3,977    5.6%    0.00013
  2000      200       3     3,764   -0.1%    0.00012
  2000      400       3     3,764   -0.1%    0.00012

Overlap is duplicated text: it is embedded, stored and — when two adjacent
chunks both surface — sent to the model twice. Sections 4 and 5 are what pay
it back. A slightly negative duplication figure is the splitter discarding
whitespace at a chunk boundary, not text going missing.

2. Embedding cache
------------------
Corpus re-ingested twice: 6 chunks, 3,977 characters
  cold:     1.3 ms,    6 embedded, $0.00013 of embedding billed
  warm:     0.1 ms,    0 embedded, $0.00000 billed
  hit rate after two passes: 50%

The hashing embedder is ~free, so the time saved here is small. The point is the
call count: on OpenAI the warm pass issues zero billable requests, and on the
local sentence-transformer it skips the same number of forward passes.

3. Query latency against corpus size
------------------------------------
  chunks  index (ms)  p50 (ms)  p95 (ms)
       6         0.6      0.16      0.17
      48         4.3      1.13      1.23
     384        34.3      8.91      9.12
    1536       138.8     35.98     36.32

Brute-force cosine over a list: latency is linear in the corpus, because every
query scores every chunk. Chroma's HNSW index is sub-linear, which is the whole
reason the default backend is Chroma and this one is for benchmarks.

4. Context assembly on the shipped corpus
-----------------------------------------
query                                             naive   kept     lean   saved
How much did revenue grow and which segment       3,747      5    3,566    4.8%
What do customers complain about most?            3,771      5    3,737    0.9%
Which modules does the platform provide?          3,747      5    3,566    4.8%

The shipped corpus is 3,977 characters in 6 chunks, so k=5
retrieves most of it and there is little to prune. That is the honest result at
this size, and it is why the next section runs the same code against a corpus
large enough for the question to matter.

5. Cost per report as k grows (synthetic 200x corpus)
-----------------------------------------------------
Corpus: 1,200 chunks, 809,140 characters

  k  naive tok  lean tok  pruned  $ before  $ after  saving
  5      1,003     1,003       0    0.0990   0.0990    0.0%
 10      1,856     1,856       0    0.1076   0.1076    0.0%
 20      3,529     2,841       0    0.1243   0.1174    5.5%
 40      6,901     2,861       0    0.1580   0.1176   25.6%
 80     13,594     2,861       0    0.2249   0.1176   47.7%

Relevance pruning shows 0 here because the hashing embedder is semantically
blind: it scores every chunk at almost the same distance, so nothing is far
enough from the best hit to drop. Against a real embedder it fires — which is
precisely why the benchmark reports it rather than assuming it. The bound that is
doing the work in this table is MAX_CONTEXT_CHARS.

'naive' is the old behaviour: concatenate every retrieved chunk, with no
relevance floor and no cap. Its prompt grows linearly with k and so does the
bill, without ever asking whether chunk 40 was worth reading. 'lean' prunes
against the best hit, trims splitter overlap and stops at MAX_CONTEXT_CHARS,
which is what makes cost per report bounded rather than a function of a knob.

6. The live pipeline (real embedder, real index)
------------------------------------------------
Index: 6 chunks, embedder sentence-transformers/all-MiniLM-L6-v2

query                                           kept  pruned   before   after
How much did revenue grow and which segment        2       3      955     486
What happened to gross margin and why?             3       2      952     573
What do customers complain about most?             2       3      835     415
What is the current Net Promoter Score?            2       3      953     534
Which modules does the platform provide?           5       0      915     874
Can the platform be self-hosted?                   3       2      916     649

Mean report prompt: 921 -> 588 tokens (36.1% less input)
Cost per report:    $0.0982 -> $0.0949 (3.4% less)

With a real embedder the distances spread out and relevance pruning does fire:
most of these queries are answered by two or three chunks of the five retrieved.
The cost saving is much smaller than the input saving because a report's 2,000
output tokens are billed at 3x the input rate and are unaffected by any of this.
```

---

## 6. The test suite

```console
$ python -m pytest -q
282 passed, 2 warnings in 2.12s
```

No test makes a network call. `tests/test_network_guard.py` proves it by
attempting one and asserting it is refused.
