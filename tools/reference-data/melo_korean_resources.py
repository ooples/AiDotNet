"""Builds the MeloTTS Korean front end's embedded resources from MeloTTS's pinned dependencies.

- g2pkk012_idioms.txt.gz, g2pkk012_table.csv.gz: g2pkk 0.1.2's (Apache-2.0) idioms.txt and table.csv, verbatim (the
  table's malformed cells are data: g2pkk reads them as they are).
- bert-kor-base-vocab.txt.gz (src/Tokenization/Resources): kykim/bert-kor-base's WordPiece vocabulary, verbatim.

Default: verify the committed files byte for byte; --write: regenerate them.
"""
import argparse
import glob
import gzip
import io
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
FRONT_END = os.path.join(ROOT, "TextToSpeech", "FrontEnd", "Resources")
TOKENIZATION = os.path.join(ROOT, "Tokenization", "Resources")


def gz(data):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        handle.write(data)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g2pkk", required=True, help="the installed g2pkk 0.1.2 package directory")
    parser.add_argument("--write", action="store_true", help="regenerate the resources (default: verify them)")
    args = parser.parse_args()
    vocab = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--kykim--bert-kor-base/snapshots/*/vocab.txt"))[0]
    outputs = {
        os.path.join(FRONT_END, "g2pkk012_idioms.txt.gz"): gz(open(os.path.join(args.g2pkk, "idioms.txt"), "rb").read()),
        os.path.join(FRONT_END, "g2pkk012_table.csv.gz"): gz(open(os.path.join(args.g2pkk, "table.csv"), "rb").read()),
        os.path.join(TOKENIZATION, "bert-kor-base-vocab.txt.gz"): gz(open(vocab, "rb").read()),
    }
    if args.write:
        for path, blob in outputs.items():
            with open(path, "wb") as handle:
                handle.write(blob)
        print(f"wrote {len(outputs)} resources")
        return
    failures = [os.path.basename(p) for p, blob in outputs.items() if open(p, "rb").read() != blob]
    if failures:
        raise SystemExit("differ: " + ", ".join(failures))
    print("all resources match")


if __name__ == "__main__":
    main()
