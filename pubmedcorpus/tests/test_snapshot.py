"""Corpus snapshot serialisation. Offline — encode/decode are pure, no DB."""

import io
import json
from datetime import date

import pytest
from pubmedcorpus.models import Abstract, Paper
from pubmedcorpus.snapshot import (ABSTRACT_COLUMNS, PAPER_COLUMNS,
                                   SnapshotFormatError, decode_record,
                                   encode_record, export, import_)

_AUTHORS = [
    {"position": 1, "is_last": False, "last_name": "Ochoa-Urrea", "fore_name": "Manuela",
     "initials": "M", "collective_name": None, "affiliation": "McGovern Medical School",
     "orcid": None, "canonical_key": "ochoa-urrea|m"},
    {"position": 2, "is_last": True, "last_name": "Lhatoo", "fore_name": "Samden D",
     "initials": "SD", "collective_name": None, "affiliation": "McGovern Medical School",
     "orcid": "0000-0001-2345-6789", "canonical_key": "lhatoo|s"},
]

_PUBMED_JSON = {
    "pmid": "40975113",
    "title": "Risk markers for SUDEP",
    "journal": "Lancet",
    "doi": "10.1016/S0140-6736(25)01636-8",
    "volume": "406",
    "abstract_text": "Background: ...",
    "mesh_terms": ["Epilepsy", "Sudden Unexpected Death in Epilepsy"],
    "authors": _AUTHORS,
}


def a_record(*, with_paper: bool = True) -> Abstract:
    record = Abstract(
        pmid="40975113",
        pubmed_xml="<PubmedArticle/>",
        pubmed_json=_PUBMED_JSON,
        entry_date=date(2025, 9, 20),
        pub_year=2025,
        pmc_id="PMC12707170",
        is_retracted=False,
        fetch_failed=False,
        paper=None,
    )
    if with_paper:
        record.paper = Paper(
            pmid="40975113",
            jats_xml="<article><body/></article>",
            fulltext_jats={
                "license": "CC BY",
                "abstract": "Background: ...",
                "sections": [{"type": "discussion", "title": "Discussion", "text": "..."}],
            },
            citation="Ochoa-Urrea M, Lhatoo SD. Risk markers for SUDEP. Lancet. 2025.",
        )
    return record


class TestEncode:
    def test_dates_become_iso_strings(self):
        # JSON has no date type; a raw date would not survive json.dumps.
        assert encode_record(a_record())["entry_date"] == "2025-09-20"

    def test_jsonb_columns_pass_through(self):
        row = encode_record(a_record())
        assert row["pubmed_json"]["mesh_terms"] == ["Epilepsy", "Sudden Unexpected Death in Epilepsy"]
        assert row["paper"]["fulltext_jats"]["sections"][0]["type"] == "discussion"

    def test_bookkeeping_columns_are_not_exported(self):
        # ingested_at / updated_at are DB-managed and regenerated on import.
        row = encode_record(a_record())
        assert "ingested_at" not in row
        assert "updated_at" not in row

    def test_the_paper_block_is_absent_without_full_text(self):
        # Its absence IS the has-full-text signal on the wire, matching the table
        # it restores into.
        assert "paper" not in encode_record(a_record(with_paper=False))

    def test_authorship_travels_inside_pubmed_json(self):
        # There is no author table to restore into, so authorship rides in the parse.
        # Ordering is the parser's, preserved verbatim rather than re-sorted on export.
        row = encode_record(a_record())
        assert "authors" not in row
        assert [a["position"] for a in row["pubmed_json"]["authors"]] == [1, 2]


class TestRoundTrip:
    def _round_trip(self, record: Abstract):
        return decode_record(json.loads(json.dumps(encode_record(record))))

    def test_survives_json_and_back(self):
        # The actual guarantee: what export writes, import can read.
        values, paper = self._round_trip(a_record())
        assert values["pmid"] == "40975113"
        assert values["entry_date"] == date(2025, 9, 20)  # parsed back to a date
        assert values["pubmed_xml"] == "<PubmedArticle/>"
        assert paper["fulltext_jats"]["sections"][0]["text"] == "..."

    def test_decode_covers_every_column(self):
        values, paper = self._round_trip(a_record())
        assert set(values) == set(ABSTRACT_COLUMNS)
        assert set(paper) == set(PAPER_COLUMNS)

    def test_no_paper_block_decodes_to_none(self):
        values, paper = self._round_trip(a_record(with_paper=False))
        assert values["pmid"] == "40975113"
        assert paper is None

    def test_null_dates_stay_null(self):
        record = a_record()
        record.entry_date = None
        values, _paper = self._round_trip(record)
        assert values["entry_date"] is None

    def test_full_text_travels(self):
        # Unlike store.py's upsert, the snapshot restores full text — it is the
        # authority on a rebuild, so dropping these would silently lose fetched JATS
        # and the raw article with it.
        assert "fulltext_jats" in PAPER_COLUMNS
        assert "jats_xml" in PAPER_COLUMNS

    def test_fetch_state_travels(self):
        # Unlike store.py's upsert, which leaves fetch_failed to pmc-fetch: here the
        # snapshot is the authority, so a restore must not re-ask PMC for thousands
        # of records it already declined.
        assert "fetch_failed" in ABSTRACT_COLUMNS

    def test_derived_values_round_trip(self):
        # citation and pubmed_json travel rather than being recomputed on import: a
        # restore must reproduce the exported corpus, not re-derive it with the
        # importing build's parser.
        values, paper = self._round_trip(a_record())
        assert paper["citation"].startswith("Ochoa-Urrea M, Lhatoo SD.")
        assert values["pubmed_json"]["authors"][1]["orcid"] == "0000-0001-2345-6789"

    def test_detail_with_no_column_survives(self):
        # The justification for so few real columns: nothing was lost, it moved into
        # the stored parse.
        values, _paper = self._round_trip(a_record())
        record = values["pubmed_json"]
        assert record["volume"] == "406"
        assert record["abstract_text"] == "Background: ..."
        assert record["authors"][0]["affiliation"] == "McGovern Medical School"
        assert record["authors"][1]["is_last"] is True


class TestOldSnapshotRejected:
    """A pre-split snapshot must not import. Reads are tolerant by design (`rec.get`),
    so without this it would land with every derived column NULL — or, once the NOT
    NULLs bite, fail with a constraint error that tells the operator nothing about
    re-exporting."""

    def _old_record(self) -> dict:
        # One flat row per paper, the shape before the abstract/paper split.
        return {
            "pmid": "40975113",
            "raw_xml": "<PubmedArticle/>",
            "abstract_text": "Background: ...",
            "fulltext_status": "ok",
            "citation": "Ochoa-Urrea M, Lhatoo SD.",
        }

    def test_flat_row_is_rejected(self):
        with pytest.raises(SnapshotFormatError, match="predates"):
            decode_record(self._old_record())

    def test_pre_citation_snapshot_is_rejected(self):
        # Older still: per-author rows and no rendered citation at all.
        with pytest.raises(SnapshotFormatError):
            decode_record({"pmid": "1", "authors": [{"position": 1}]})

    def test_the_error_names_the_record_and_the_fix(self):
        with pytest.raises(SnapshotFormatError) as exc:
            decode_record(self._old_record())
        assert "40975113" in str(exc.value)
        assert "Re-export" in str(exc.value)


# There is no restore marker any more. A snapshot's records carry their own
# `entry_date` (asserted in TestRoundTrip above), and that alone is what lets a later
# sync resume incrementally — its cursor is `max(entry_date)` over whatever the corpus
# actually holds, restored or freshly pulled makes no difference. See
# `pubmedcorpus.sync._cursor` and test_sync.py for the cursor itself.


# --- export / import_ ---------------------------------------------------------------
#
# **Untested until session injection made them reachable.** Both used to open their own
# `session_scope`, so only the pure encode/decode pair above could be exercised offline. These cover
# what the streaming halves decide, which is separable from the serialisation the tests above pin.


class _ExportSession:
    """Returns canned records for the one SELECT `export` issues."""

    def __init__(self, records):
        self._records = records

    def execute(self, _stmt):
        return _Scalars(self._records)


class _Scalars:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self._rows


class _ImportSession:
    """Counts commits and collects the upserts `_flush` emits."""

    def __init__(self):
        self.statements = []
        self.commits = 0

    def execute(self, statement):
        self.statements.append(statement)

    def commit(self):
        self.commits += 1


class TestExport:
    def test_one_json_line_per_record(self):
        out = io.StringIO()
        n = export(_ExportSession([a_record(), a_record()]), out)
        assert n == 2
        assert len(out.getvalue().strip().split("\n")) == 2

    def test_each_line_is_the_encoded_record(self):
        """`export` must not re-implement serialisation — it delegates to `encode_record`."""
        out = io.StringIO()
        export(_ExportSession([a_record()]), out)
        assert json.loads(out.getvalue()) == encode_record(a_record())

    def test_a_record_without_full_text_omits_the_nested_block(self):
        """Absence *is* the has-full-text signal on the wire, matching the table it restores into."""
        out = io.StringIO()
        export(_ExportSession([a_record(with_paper=False)]), out)
        assert "paper" not in json.loads(out.getvalue())

    def test_an_empty_corpus_writes_nothing(self):
        out = io.StringIO()
        assert export(_ExportSession([]), out) == 0
        assert out.getvalue() == ""


class TestImport:
    def _lines(self, count: int) -> list[str]:
        rec = encode_record(a_record())
        return [json.dumps({**rec, "pmid": str(i)}) for i in range(count)]

    def test_returns_the_number_imported(self):
        session = _ImportSession()
        assert import_(session, self._lines(3)) == 3

    def test_blank_lines_are_skipped_not_counted(self):
        """A trailing newline in a snapshot file must not read as a record."""
        session = _ImportSession()
        lines = self._lines(2)
        assert import_(session, [lines[0], "", "  ", lines[1], ""]) == 2

    def test_it_commits_per_batch_and_once_for_the_tail(self):
        """**What makes a partial import usable.** Five records at `batch_size=2` is two full batches
        plus a tail of one — three commits. Committing only at the end would mean an interrupted import
        left nothing, and the CLI's error path documents relying on the opposite.
        """
        session = _ImportSession()
        assert import_(session, self._lines(5), batch_size=2) == 5
        assert session.commits == 3

    def test_an_exact_multiple_of_the_batch_size_does_not_double_commit(self):
        """The tail branch must not fire on an empty leftover batch."""
        session = _ImportSession()
        assert import_(session, self._lines(4), batch_size=2) == 4
        assert session.commits == 2

    def test_a_record_with_full_text_upserts_both_tables(self):
        session = _ImportSession()
        import_(session, self._lines(1))
        assert len(session.statements) == 2

    def test_a_record_without_full_text_upserts_only_the_abstract(self):
        """Import is additive: a record that lost its full text is not un-promoted, because deleting
        a paper row would take everything keyed to it along via FK cascade."""
        session = _ImportSession()
        import_(session, [json.dumps(encode_record(a_record(with_paper=False)))])
        assert len(session.statements) == 1

    def test_a_pre_split_snapshot_is_rejected_before_anything_commits(self):
        """The CLI promises an old file "trips this on its first record, before any batch is
        committed, so nothing lands". This is that promise as a test."""
        session = _ImportSession()
        with pytest.raises(SnapshotFormatError):
            import_(session, [json.dumps({"pmid": "1", "raw_xml": "<x/>"})])
        assert session.commits == 0
