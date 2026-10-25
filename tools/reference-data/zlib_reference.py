"""zlib reference: Python's zlib.compress(data) (zlib 1.3.2, level 6) on a corpus, for the ZlibDeflate port.

Inputs: empty, short and repetitive text, UTF-8 text in several scripts, random bytes, text longer than 64 KB (the
window slides) and long enough to fill the 16K-symbol buffer (several blocks), and Whisper-like repeated transcripts.
Writes tests/AiDotNet.Tests/Helpers/ReferenceData/zlib_reference.json (inputs and outputs in base64). Default: verify;
--write: regenerate.
"""
import argparse
import base64
import random
import sys
import zlib

import _fixture

FIXTURE = "tests/AiDotNet.Tests/Helpers/ReferenceData/zlib_reference.json"


def corpus():
    rng = random.Random(7)
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "speech", "model", "audio", "token"]
    cases = [b"", b"a", b"hello", b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", " bidding".encode() * 300,
             "雪".encode() * 400, "Hello, world! This is a test of the compression ratio.".encode(),
             "Привет мир, 日本語のテキスト, ελληνικά, עברית".encode() * 20,
             bytes(rng.getrandbits(8) for _ in range(5000)),
             " ".join(rng.choice(words) for _ in range(30000)).encode(),
             bytes(rng.choice(b"ab") for _ in range(70000)),
             bytes(rng.getrandbits(8) for _ in range(70000))]
    return [{"input": base64.b64encode(c).decode(), "output": base64.b64encode(zlib.compress(c)).decode()} for c in cases]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if zlib.ZLIB_RUNTIME_VERSION != "1.3.2":
        raise SystemExit("needs zlib 1.3.2, found " + zlib.ZLIB_RUNTIME_VERSION)
    data = {"zlib": zlib.ZLIB_RUNTIME_VERSION, "cases": corpus()}
    if args.write:
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE)
        return
    same = _fixture.load(FIXTURE)["cases"] == data["cases"]
    print("zlib: match" if same else "MISMATCH")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
