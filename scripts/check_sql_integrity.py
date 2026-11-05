"""Verify 001–005 SQL DDL matches the user's unmodified source snapshot."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
hashes = {}
for line in (root / "docs/SQL_SHA256.txt").read_text().splitlines():
    name, expected = line.split('  ', 1)
    actual = hashlib.sha256((root / 'sql' / name).read_bytes()).hexdigest()
    hashes[name] = actual == expected
for name, ok in sorted(hashes.items()):
    print(f'{"PASS" if ok else "FAIL"} {name}')
if not all(hashes.values()):
    raise SystemExit(1)
