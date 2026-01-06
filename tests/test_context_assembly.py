"""
Tests for context assembly.

Every character this module emits is billed as an input token on every query,
so the reductions it performs are the project's main cost lever. Each of them
is tested for both halves of its contract: that it removes what it should, and
that it never removes something the report then cannot be written from.
"""

import pytest

from src.agent.context import (
    Citation,
    assemble_context,
    prune_by_relevance,
    source_header,
)


class Doc:
    """The slice of langchain's Document that assembly reads."""

    def __init__(self, page_content, source=None):
        self.page_content = page_content
        self.metadata = {"source": source} if source else {}


def scored(*pairs):
    return [(Doc(text, source), score) for text, source, score in pairs]


class TestRelevancePruning:
    def test_chunks_far_worse_than_the_best_hit_are_dropped(self):
        """
        Retrieval returns exactly k chunks whether or not k are relevant. A
        question answered by one paragraph still paid for four more.
        """
        rows = [("a", 0.10), ("b", 0.12), ("c", 0.90)]

        kept, pruned = prune_by_relevance(rows, margin=0.35)

        assert [text for text, _ in kept] == ["a", "b"]
        assert pruned == 1

    def test_a_margin_of_zero_keeps_everything(self):
        rows = [("a", 0.1), ("b", 5.0)]

        kept, pruned = prune_by_relevance(rows, margin=0.0)

        assert len(kept) == 2
        assert pruned == 0

    def test_the_best_hit_is_never_pruned(self):
        """A query must not end up with no context because of this rule."""
        rows = [("a", 0.5), ("b", 9.0)]

        kept, _ = prune_by_relevance(rows, margin=0.01)

        assert kept[0][0] == "a"

    def test_a_retriever_that_reports_no_scores_is_not_pruned(self):
        rows = [("a", None), ("b", None)]

        kept, pruned = prune_by_relevance(rows, margin=0.35)

        assert len(kept) == 2
        assert pruned == 0

    def test_similarity_scores_are_declined_rather_than_pruned_backwards(self):
        """
        Some backends report similarity, where higher is better. Pruning with
        the wrong polarity would drop exactly the best hits, so the rule
        declines to act rather than guessing.
        """
        rows = [("best", -0.9), ("worst", -0.1)]

        kept, pruned = prune_by_relevance(rows, margin=0.35)

        assert len(kept) == 2
        assert pruned == 0

    def test_an_exact_match_does_not_collapse_the_context_to_one_chunk(self):
        """
        A distance of 0 makes a purely multiplicative ceiling 0 too, which
        would drop every other chunk however good it was.
        """
        rows = [("a", 0.0), ("b", 0.2), ("c", 0.9)]

        kept, _ = prune_by_relevance(rows, margin=0.35)

        assert [text for text, _ in kept] == ["a", "b"]


class TestOverlapDeduplication:
    def test_splitter_overlap_shared_with_an_earlier_chunk_is_trimmed(self):
        """
        The splitter repeats CHUNK_OVERLAP characters between adjacent chunks
        so a sentence is not cut in half. When retrieval returns both — which
        is common, because adjacent chunks are about the same thing — that
        overlap was sent to the model twice and billed twice.
        """
        tail = "the northern region drove enterprise renewals all year. "
        first = "Revenue rose twenty two percent. " + tail
        second = tail + "Margin improved by four points."

        bundle = assemble_context(
            scored((first, "a.md", 0.1), (second, "a.md", 0.11)),
            max_chars=10_000,
            chunk_overlap=len(tail),
        )

        assert bundle.overlap_chars_trimmed == len(tail)
        assert bundle.text.count(tail.strip()) == 1

    def test_an_identical_chunk_is_dropped_entirely(self):
        bundle = assemble_context(
            scored(("exactly the same text", "a.md", 0.1), ("exactly the same text", "a.md", 0.1)),
            max_chars=10_000,
            chunk_overlap=200,
        )

        assert bundle.dropped_as_duplicate == 1
        assert len(bundle.citations) == 1

    def test_identical_text_from_two_different_files_is_kept_twice(self):
        """
        Two documents can legitimately share a paragraph — a boilerplate legal
        notice, a repeated table header. Collapsing them would lose a source
        the report is entitled to cite.
        """
        bundle = assemble_context(
            scored(("shared boilerplate", "a.md", 0.1), ("shared boilerplate", "b.md", 0.1)),
            max_chars=10_000,
            chunk_overlap=200,
        )

        assert bundle.dropped_as_duplicate == 0
        assert [c.source for c in bundle.citations] == ["a.md", "b.md"]

    def test_a_short_coincidental_overlap_is_left_alone(self):
        """
        Two chunks sharing "The " is coincidence, not splitter overlap, and
        trimming it would cut a word off the front of a sentence.
        """
        bundle = assemble_context(
            scored(
                ("Margin improved. The", "a.md", 0.1), ("The rest of it entirely.", "a.md", 0.1)
            ),
            max_chars=10_000,
            chunk_overlap=200,
        )

        assert bundle.overlap_chars_trimmed == 0


class TestBudget:
    def test_the_context_never_exceeds_the_budget(self):
        bundle = assemble_context(
            scored(*[("x" * 500, f"doc{i}.md", 0.1) for i in range(10)]),
            max_chars=300,
        )

        assert len(bundle.text) <= 300
        assert bundle.truncated

    def test_everything_fits_when_there_is_room(self):
        bundle = assemble_context(
            scored(*[("y" * 100, f"doc{i}.md", 0.1) for i in range(5)]),
            max_chars=100_000,
        )

        assert len(bundle.citations) == 5
        assert not bundle.truncated

    def test_no_documents_yields_an_empty_bundle_rather_than_an_error(self):
        bundle = assemble_context([], max_chars=1000)

        assert bundle.text == ""
        assert bundle.citations == []


class TestCitations:
    def test_every_kept_chunk_gets_a_sequential_marker(self):
        bundle = assemble_context(
            scored(("alpha text here", "a.md", 0.1), ("beta text here", "b.md", 0.2)),
            max_chars=10_000,
        )

        assert [c.id for c in bundle.citations] == ["S1", "S2"]

    def test_the_marker_in_the_text_matches_the_citation_record(self):
        bundle = assemble_context(scored(("alpha", "a.md", 0.1)), max_chars=10_000)

        assert source_header(1, "a.md") in bundle.text
        assert bundle.citations[0].id == "S1"

    def test_markers_stay_contiguous_when_a_chunk_is_dropped(self):
        """
        Numbering off the loop index instead of the accepted count would leave
        a gap — a prompt with [S1] and [S3] and no [S2], which invites a model
        to cite the missing one.
        """
        bundle = assemble_context(
            scored(
                ("alpha text here", "a.md", 0.1),
                ("alpha text here", "a.md", 0.1),
                ("gamma text here", "c.md", 0.1),
            ),
            max_chars=10_000,
            chunk_overlap=200,
        )

        assert [c.id for c in bundle.citations] == ["S1", "S2"]

    def test_the_retrieval_distance_travels_with_the_citation(self):
        bundle = assemble_context(scored(("alpha", "a.md", 0.42)), max_chars=10_000)

        assert bundle.citations[0].score == pytest.approx(0.42)

    def test_a_chunk_without_source_metadata_still_gets_a_label(self):
        bundle = assemble_context([(Doc("alpha"), 0.1)], max_chars=10_000)

        assert bundle.sources == ["Document 1"]

    def test_the_citation_excerpt_is_bounded(self):
        """`as_dict` goes into an API response; a whole chunk does not belong there."""
        citation = Citation(id="S1", source="a.md", text="z" * 5000)

        assert len(citation.as_dict()["excerpt"]) == 280


class TestAccounting:
    def test_the_bundle_reports_what_it_saved(self):
        tail = "a repeated tail sentence that both chunks contain in full. "
        bundle = assemble_context(
            scored(("head. " + tail, "a.md", 0.1), (tail + "tail.", "a.md", 0.11)),
            max_chars=10_000,
            chunk_overlap=len(tail),
        )

        assert bundle.raw_chars > len(bundle.text) - len(tail)
        assert bundle.overlap_chars_trimmed > 0

    def test_ordering_is_preserved(self):
        """The retriever returns its best match first and that order is meaningful."""
        bundle = assemble_context(
            scored(("first text", "a.md", 0.1), ("second text", "b.md", 0.2)),
            max_chars=10_000,
        )

        assert bundle.text.index("first text") < bundle.text.index("second text")
