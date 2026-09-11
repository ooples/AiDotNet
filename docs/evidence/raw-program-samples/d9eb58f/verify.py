"""Verify archived receipts and exact runtime/source bytes; no extraction or execution."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET
import zipfile

root = Path(__file__).resolve().parent
revision = "d9eb58f46e1f97c3bb300dea0c5e2b70f1321964"


def require(condition, message):
    if not condition:
        raise ValueError(message)


integrity = json.loads((root / "integrity.json").read_bytes())
source = root / "verification.zip"
require(integrity["SourceRevision"] == revision and source.stat().st_size == integrity["ArchiveBytes"] < 100 * 1024 * 1024, "Identity/size mismatch.")
with source.open("rb") as stream:
    require(hashlib.file_digest(stream, "sha256").hexdigest() == integrity["ArchiveSha256"], "Archive checksum mismatch.")
with zipfile.ZipFile(source) as archive:
    names = archive.namelist()
    require(len(names) == len(set(names)) == integrity["Entries"] < 100, "Duplicate/excessive archive entries.")
    require(all(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts and "\\" not in name and ":" not in name for name in names), "Unsafe archive path.")
    require(all(entry.file_size <= 64 * 1024 * 1024 for entry in archive.infolist()) and sum(entry.file_size for entry in archive.infolist()) <= 512 * 1024 * 1024, "Archive exceeds bounds.")
    manifest = json.loads(archive.read("manifest.json"))
    require(manifest["SourceRevision"] == revision and manifest["CoreSourceRevision"] == "255feb24369702a32ea9db7a3f8a0b7a847d2762" and
        set(manifest["Files"]) == set(names) - {"manifest.json"}, "Manifest mismatch.")
    for name, checksum in manifest["Files"].items():
        require(hashlib.sha256(archive.read(name)).hexdigest() == checksum, "Artifact checksum mismatch: " + name)
    for filename, expected in (("all-evolution-pinned-net10.trx", 963), ("all-evolution-net8.trx", 963), ("raw-evidence-net471.trx", 22)):
        counters = ET.fromstring(archive.read("tests/" + filename)).find(".//{*}Counters")
        require(counters is not None and all(counters.attrib[key] == str(expected) for key in ("total", "executed", "passed")) and counters.attrib["failed"] == "0", "Test receipt mismatch.")
    for framework in ("net10.0", "net8.0", "net471"):
        for filename in ("AiDotNet.dll", "AiDotNet.Evolution.dll"):
            require(f"runtime/{framework}/{filename}" in manifest["Files"], "Missing tested runtime.")
print(json.dumps({"Verified": True, "SourceRevision": revision, "ModernIntegrationTestsEach": 963, "LegacyProviderTests": 22}))
