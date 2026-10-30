"""SpiritLM tokenizer reference: the release's tokenizer is LLaMA's SentencePiece model with the markers and speech tokens
added as user-defined pieces after the 32,000 text pieces ([Text], [Speech], [Hu0]..[Hu500], then [Pi0]..[Pi63] and
[St0]..[St99] for Expressive), read by transformers' LlamaTokenizer (legacy, add_bos_token): <s> and SentencePiece's
encoding of the whole string.

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/spiritlm_tokenizer_reference.json:
  - a tiny BPE model laid out the same way (its user-defined pieces moved after the text pieces), for the test fixture,
    with its encodings of a few prompts;
  - with --real (the released spirit-lm-expressive-7b/tokenizer.model), that model's encodings and decodings of the same
    prompts, for a test gated on the file;
  - with --spiritlm (the release's checkout), its arbitrary-output helpers on a few strings: does_start_with_speech_token,
    does_end_with_speech_token, find_prompt_last_speech_start_position and Spiritlm._parse_speech_and_text.
Default: verify; --write: regenerate.
"""
import argparse
import base64
import os
import sys
import tempfile

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as model_pb2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/spiritlm_tokenizer_reference.json"
UNITS, PITCH, STYLE = 12, 4, 3
CORPUS = ["the cat sat on the mat", "hello there", "the largest country in the world is", "here is a story about a llama",
          "one two three four five", "she said turning her head", "are you really going to do that", "eat your raisins outdoors"]
PROMPTS = ["[Text]hello there[Speech]", "[Speech][Hu3][Hu7][Hu3][Text]the cat", "[Speech][St2][Pi1][Hu0][Hu11][Pi3][Hu5]",
           "[Text]the mat [Speech][Hu1][Text] one two", "[Hu12][Text]zebra?"]


def user_pieces():
    return (["[Text]", "[Speech]"] + [f"[Hu{u}]" for u in range(UNITS + 1)] + [f"[Pi{p}]" for p in range(PITCH)]
            + [f"[St{s}]" for s in range(STYLE)])


def tiny_model():
    with tempfile.TemporaryDirectory() as folder:
        corpus = os.path.join(folder, "corpus.txt")
        with open(corpus, "w", encoding="utf-8") as handle:
            handle.write("\n".join(CORPUS * 20))
        prefix = os.path.join(folder, "tiny")
        spm.SentencePieceTrainer.train(input=corpus, model_prefix=prefix, vocab_size=96, model_type="bpe", character_coverage=1.0,
                                       user_defined_symbols=user_pieces(), bos_id=1, eos_id=2, unk_id=0, pad_id=-1,
                                       add_dummy_prefix=True, minloglevel=2)
        proto = model_pb2.ModelProto()
        with open(prefix + ".model", "rb") as handle:
            proto.ParseFromString(handle.read())
    # As the release: the user-defined pieces after the text pieces.
    pieces = list(proto.pieces)
    user = [p for p in pieces if p.type == model_pb2.ModelProto.SentencePiece.USER_DEFINED]
    rest = [p for p in pieces if p.type != model_pb2.ModelProto.SentencePiece.USER_DEFINED]
    del proto.pieces[:]
    for p in rest + user:
        proto.pieces.add().CopyFrom(p)
    return proto.SerializeToString()


def encodings(processor):
    return [{"text": t, "ids": [1] + processor.encode(t), "decoded": processor.decode(processor.encode(t))} for t in PROMPTS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", help="the released tokenizer.model")
    parser.add_argument("--spiritlm", help="the facebookresearch/spiritlm checkout")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    stored = None if args.write else _fixture.load(FIXTURE)
    blob = base64.b64decode(stored["tiny_model_base64"]) if stored else tiny_model()
    tiny = spm.SentencePieceProcessor(model_proto=blob)
    data = {"tiny_pieces": tiny.get_piece_size(), "tiny_text_marker": tiny.piece_to_id("[Text]"), "tiny": encodings(tiny)}
    if args.spiritlm:
        sys.path.insert(0, args.spiritlm)
        # spiritlm.model imports the speech tokenizers (torchaudio, fairseq-free HuBERT) on import; utils alone is enough.
        import importlib.util
        spec = importlib.util.spec_from_file_location("spiritlm_utils", os.path.join(args.spiritlm, "spiritlm", "model", "utils.py"))
        utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(utils)
        source = open(os.path.join(args.spiritlm, "spiritlm", "model", "spiritlm_model.py"), encoding="utf-8").read()
        start = source.index("    def _parse_speech_and_text(")
        end = source.index("    def _decode_from_generated_output(")
        namespace = {}
        body = 'class Spiritlm:\n    TEXT_PROMPT_PREFIX = "[Text]"\n    SPEECH_PROMPT_PREFIX = "[Speech]"\n' + source[start:end]
        exec(body, namespace)
        parse = namespace["Spiritlm"]._parse_speech_and_text
        cases = ["[Speech][Hu3][Hu7]", "[Hu3][Hu5]", "[Text]hello[Hu1]", "abc", "[Pi12]x", "[St5][Hu1][Text]the cat [Speech][Hu2]",
                 "[Speech][Hu1][Hu2][Text]x[Speech][Hu3][Hu4][Hu5]", "[Hu12345678]", "text [weird] [Hu4]tail[Text][Speech]"]
        data["helper_cases"] = [{"text": c, "starts": utils.does_start_with_speech_token(c), "ends": utils.does_end_with_speech_token(c),
                                 "last_start": utils.find_prompt_last_speech_start_position(c),
                                 "split": [[chunk, kind] for chunk, kind in parse(None, c)]} for c in cases]
    if args.real:
        real = spm.SentencePieceProcessor(model_file=args.real)
        data.update(real_pieces=real.get_piece_size(), real=encodings(real))
    if args.write:
        _fixture.save(FIXTURE, {"tiny_model_base64": base64.b64encode(blob).decode("ascii"), "units": UNITS, "pitch": PITCH, "style": STYLE, **data}, __file__)
        print("wrote", FIXTURE, data["tiny_pieces"], data["tiny_text_marker"], data["tiny"][0])
        return
    check = _fixture.Comparison("SpiritLM tokenizer")
    for k, v in data.items():
        if k in stored and stored[k] != v:
            check.failures.append(f"{k}: differs")
    check.report(exact=True)


if __name__ == "__main__":
    main()
