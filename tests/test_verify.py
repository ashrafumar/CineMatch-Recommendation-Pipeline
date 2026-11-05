"""Stage 1 unit tests. CineMatch | Roll Number: 2025AG05135."""
import csv
import hashlib
from pathlib import Path
import pytest
from cinematch.cli import make_parser
from cinematch.ingest import SOURCES
from cinematch.verify import EXPECTED_COUNTS, EXPECTED_MD5, HEADERS, md5_file, verify_file, read_manifest


def test_source_baselines():
    assert EXPECTED_COUNTS["ratings.csv"] == 32_000_204
    assert EXPECTED_COUNTS["movies.csv"] == EXPECTED_COUNTS["links.csv"] == 87_585
    assert EXPECTED_COUNTS["tags.csv"] == 2_000_072
    assert set(SOURCES) == {"movies", "links", "ratings"}


def test_pinned_hash_format():
    assert all(len(value) == 32 and all(c in "0123456789abcdef" for c in value)
               for value in EXPECTED_MD5.values())


def test_matching_fixture(tmp_path):
    file = tmp_path / "ratings.csv"
    file.write_text("userId,movieId,rating,timestamp\n1,2,4.5,100\n", encoding="utf-8")
    expected = hashlib.md5(file.read_bytes()).hexdigest()
    assert verify_file(file, expected, HEADERS["ratings.csv"]) == expected


def test_md5_mismatch(tmp_path):
    file = tmp_path / "ratings.csv"
    file.write_text("userId,movieId,rating,timestamp\n", encoding="utf-8")
    with pytest.raises(ValueError, match="MD5 mismatch"):
        verify_file(file, "0" * 32, HEADERS["ratings.csv"])


def test_bad_header(tmp_path):
    file = tmp_path / "ratings.csv"
    file.write_text("wrong,columns\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected CSV header"):
        verify_file(file, md5_file(file), HEADERS["ratings.csv"])


def test_manifest(tmp_path):
    (tmp_path / "checksums.txt").write_text("abcd file.csv\n", encoding="utf-8")
    assert read_manifest(tmp_path) == {"file.csv": "abcd"}


def test_cli_commands():
    parser = make_parser()
    assert parser.parse_args(["ingest", "--kind", "ratings"]).kind == "ratings"
    assert parser.parse_args(["--source-dir", "/tmp/ml-32m", "verify"]).source_dir == "/tmp/ml-32m"
    assert parser.parse_args(["reconcile"]).command == "reconcile"
