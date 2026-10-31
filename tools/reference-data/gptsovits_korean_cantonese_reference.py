"""Reference fixture for GPT-SoVITS's Korean and Cantonese front ends, and the Cantonese dictionary resource.

Runs GPT-SoVITS's own text package (RVC-Boss/GPT-SoVITS at b4570b8) in the gptsovits-ref image (g2pk2 0.0.3 with
python-mecab-ko 1.3.7, jamo 0.4.1, ToJyutping 3.2.0), from the GPT-SoVITS checkout:

    docker run --rm -v <refsrc>:/ref -v <repo>:/repo -w /ref/GPT-SoVITS gptsovits-ref \
        python /repo/tools/reference-data/gptsovits_korean_cantonese_reference.py [--write-resources]

Records korean.g2p and cantonese.text_normalize / g2p of sentences (and clean_text's phones). --write-resources also
writes tojyutping_words.tsv.gz (every word of ToJyutping's trie with its first reading, one syllable per character:
"word<TAB>syllable ...") and g2pk2003_table.csv.gz (g2pk2's table.csv byte for byte); without it they are verified.

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/gptsovits_korean_cantonese_reference.json.
"""
import gzip
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESOURCES = os.path.join(HERE, "..", "..", "src", "TextToSpeech", "FrontEnd", "Resources")
sys.path.insert(0, os.path.join(os.getcwd(), "GPT_SoVITS"))
os.environ.setdefault("version", "v2")

from text import korean, cantonese, cleaner  # noqa: E402

KOREAN = [
    "안녕하세요",
    "나의 친구가 mp3 file 3개를 다운받고 있다",
    "오늘 날씨가 정말 좋네요!",
    "저는 학교에 갑니다. 그리고 공부를 합니다.",
    "읽고 있는 책이 재미있어요?",
    "할 일이 많아서 바빠요, 그래도 괜찮아요.",
    "꽃잎이 떨어지는 봄날에 우리는 만났다.",
    "1,234원과 25살, 3마리의 고양이.",
    "AI와 IT, KOREA 그리고 CEO.",
    "닭이 울고 밟고 넓은 들에서 놀 래요.",
    "의사의 의견에 동의합니다.",
    "가을 로 갈 래 서울 로",
    "값이 싸요; 없어요… 정말？",
    "희망과 의지, 무늬와 띄어쓰기",
    "끝",
    "ㄱ",
]

CANTONESE = [
    "佢個鋤頭太短啦。",
    "你好，我係香港人！",
    "今日天氣好好，我哋去飲茶啦。",
    "呢個價錢係123蚊，好平呀？",
    "唔該晒，多謝你幫手。",
    "我哋2024年3月5日見面，大約20至30個人。",
    "「真係？」佢問我。",
    "銀行行長重新講咗一次。",
    "你食咗飯未呀？食咗喇！",
    "香港嘅夜景好靚……",
]


def gz(data):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        handle.write(data)
    return buffer.getvalue()


def tojyutping_words():
    from ToJyutping.Trie import root
    lines = []

    def walk(node, key):
        if node.v:
            first = node.v[0]
            syllables = [first.jyutping] if len(key) == 1 else [j.jyutping for j in first]
            if len(syllables) != len(key):
                raise ValueError(f"{key!r} has {len(syllables)} syllables")
            lines.append(f"{key}\t{' '.join(syllables)}\n")
        for c in sorted(node):
            walk(node[c], key + c)

    walk(root, "")
    return gz("".join(lines).encode("utf-8"))


def main():
    import g2pk2
    with open(os.path.join(os.path.dirname(g2pk2.__file__), "table.csv"), "rb") as handle:
        table = gz(handle.read())
    resources = {"tojyutping_words.tsv.gz": tojyutping_words(), "g2pk2003_table.csv.gz": table}
    for name, data in resources.items():
        path = os.path.join(RESOURCES, name)
        if "--write-resources" in sys.argv:
            with open(path, "wb") as handle:
                handle.write(data)
            print(f"wrote {name} ({len(data)} bytes)")
        else:
            with open(path, "rb") as handle:
                if handle.read() != data:
                    raise SystemExit(f"{name} differs")

    korean_out = [{"text": t, "phones": korean.g2p(t), "clean": cleaner.clean_text(t, "ko", "v2")[0]} for t in KOREAN]
    cantonese_out = []
    for t in CANTONESE:
        norm = cantonese.text_normalize(t)
        phones, word2ph = cantonese.g2p(norm)
        cantonese_out.append({"text": t, "normalized": norm, "phones": phones, "word2ph": word2ph,
                              "clean": cleaner.clean_text(t, "yue", "v2")[0]})
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData",
                        "gptsovits_korean_cantonese_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"korean": korean_out, "cantonese": cantonese_out}, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(f"wrote {len(korean_out)} Korean and {len(cantonese_out)} Cantonese sentences")


if __name__ == "__main__":
    main()
