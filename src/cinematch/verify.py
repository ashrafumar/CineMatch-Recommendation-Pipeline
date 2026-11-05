"""MovieLens manifest verification. CineMatch | Roll Number: 2025AG05135."""
import csv
import hashlib
from pathlib import Path

# Expected exact counts confirmed from the downloaded MovieLens 32M CSV files.
EXPECTED_COUNTS = {
    "ratings.csv": 32_000_204,
    "movies.csv": 87_585,
    "links.csv": 87_585,
    "tags.csv": 2_000_072,
}
EXPECTED_MD5 = {
    "links.csv": "8f033867bcb4e6be8792b21468b4fa6e",
    "movies.csv": "0df90835c19151f9d819d0822e190797",
    "ratings.csv": "cf12b74f9ad4b94a011f079e26d4270a",
    "tags.csv": "963bf4fa4de6b8901868fddd3eb54567",
}
HEADERS = {
    "ratings.csv": ["userId", "movieId", "rating", "timestamp"],
    "movies.csv": ["movieId", "title", "genres"],
    "links.csv": ["movieId", "imdbId", "tmdbId"],
    "tags.csv": ["userId", "movieId", "tag", "timestamp"],
}


def md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute a streaming MD5 digest without loading the entire CSV into memory."""
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected_md5: str, expected_header: list[str]) -> str:
    """Check a source CSV against the pinned MD5 checksum and required header."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    actual = md5_file(path)
    if actual.lower() != expected_md5.lower():
        raise ValueError(f"MD5 mismatch for {path.name}: expected {expected_md5}, got {actual}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    if header != expected_header:
        raise ValueError(f"Unexpected CSV header in {path.name}: {header!r}; expected {expected_header!r}")
    return actual


def verify_dataset(directory: Path, include_tags: bool = True) -> dict[str, str]:
    """Verify the official MovieLens files before any database ingestion."""
    files = EXPECTED_MD5 if include_tags else {k: v for k, v in EXPECTED_MD5.items() if k != "tags.csv"}
    return {filename: verify_file(directory / filename, checksum, HEADERS[filename])
            for filename, checksum in files.items()}


def read_manifest(directory: Path) -> dict[str, str]:
    """Optional inspection helper; never trust this alone instead of pinned hashes."""
    result = {}
    for line in (directory / "checksums.txt").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split()
        result[filename] = digest
    return result
