"""Reference fixture for Llasa's token handling (Ye et al. 2025; zhenye234/LLaSA_training and the HKUSTAudio/Llasa-1B model
card), outside the networks, with Llasa-1B's own tokenizer:

* tokenizer: encodings of strings mixing text, Llama 3's special tokens and Llasa's markers and speech tokens;
* prompts: the model card's chat (user "Convert the text to speech:" with the text between the text-understanding
  markers, assistant opened with the speech-generation marker and, for a speech prompt, its codes as speech tokens)
  through tokenizer.apply_chat_template(tokenize=True, continue_final_message=True), with a fixed date_string;
* training: get_memmap_from_token.process_audio_id (data_instruction/) makes the stored row of a transcript and its codes,
  and train_tts.TTSDataset.__getitem__ turns it into input_ids, labels and the attention mask (both functions run from the repository's own
  source, extracted with ast, and the chat template's date fixed; transformers 5 has no encode_plus, so it is routed to
  __call__, which it was for one text, and its apply_chat_template's result is read back as the list of ids 4.46 gave);
* reading codes: the model card's extract_speech_ids (copied verbatim below) over tokenizer.batch_decode(ids,
  skip_special_tokens=True) of a generated stream mixing speech tokens, text tokens and special tokens (each id decoded
  on its own, as batch_decode did for a 1-D tensor in transformers 4.x).

Usage: python llasa_reference.py. Needs ../refsrc/LLaSA_training (zhenye234/LLaSA_training at 479acd5), ../refsrc/llasa-hf
(HKUSTAudio/Llasa-1B's tokenizer files at cef8ce6), numpy, torch and transformers.
Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/llasa_reference.json.
"""
import ast
import json
import os
import random
import tempfile

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
REFSRC = os.path.join(HERE, "..", "..", "..", "refsrc")
TRAINING = os.path.join(REFSRC, "LLaSA_training")
DATE = "05 Mar 2025"


# ---- the model card (verbatim) ------------------------------------------------------------------------------------
def ids_to_speech_tokens(speech_ids):

    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str


def extract_speech_ids(speech_tokens_str):

    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids


def card_prompt(tokenizer, text, prefix_codes):
    formatted_text = f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"
    chat = [
        {"role": "user", "content": "Convert the text to speech:" + formatted_text},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(ids_to_speech_tokens(prefix_codes))}
    ]
    out = tokenizer.apply_chat_template(chat, tokenize=True, continue_final_message=True, date_string=DATE)
    return list(out["input_ids"] if hasattr(out, "keys") else out)


# ---- the training repository's own functions ----------------------------------------------------------------------
def extract(path, name):
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    return ast.get_source_segment(open(path, encoding="utf-8").read(), node)


def main():
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(REFSRC, "llasa-hf"))
    # The chat template's date comes from strftime_now when no date_string is passed; fix it. transformers 5 returns a
    # BatchEncoding where the 4.46 the recipe ran on returned the list of ids; return the list.
    template = tokenizer.apply_chat_template

    def apply_chat_template(chat, **kwargs):
        kwargs.setdefault("date_string", DATE)
        out = template(chat, **kwargs)
        return list(out["input_ids"]) if hasattr(out, "keys") else out

    tokenizer.apply_chat_template = apply_chat_template
    rng = random.Random(5)

    encodings = []
    for text in ["Hello there.", "<|TEXT_UNDERSTANDING_START|>Hi <|s_17|><|s_65535|> mixed<|eot_id|> text",
                 "突然，身边一阵笑声。", "<|SPEECH_GENERATION_START|><|s_0|><|s_1234|><|SPEECH_GENERATION_END|>"]:
        encodings.append({"text": text, "ids": tokenizer.encode(text, add_special_tokens=False)})

    prompts = []
    for text, prefix in [("Dealing with family secrets is never easy.", []), ("  Spaces around, and a tab.\t", []),
                         ("对，这就是我万人敬仰的太乙真人。 Then English.", [rng.randrange(65536) for _ in range(7)])]:
        prompts.append({"text": text, "prefix": prefix, "ids": card_prompt(tokenizer, text, prefix)})

    namespace = {"os": os, "np": np, "torch": torch}
    exec(extract(os.path.join(TRAINING, "data_instruction", "get_memmap_from_token.py"), "init_worker"), namespace)
    exec(extract(os.path.join(TRAINING, "data_instruction", "get_memmap_from_token.py"), "process_audio_id"), namespace)
    exec("from torch.utils.data import Dataset\n" + extract(os.path.join(TRAINING, "train_tts.py"), "TTSDataset"), namespace)
    special_tokens = ['<|TEXT_GENERATION_START|>', '<|TEXT_GENERATION_END|>', '<|TEXT_UNDERSTANDING_START|>',
                      '<|TEXT_UNDERSTANDING_END|>', '<|SPEECH_GENERATION_START|>', '<|SPEECH_GENERATION_END|>',
                      '<|SPEECH_UNDERSTANDING_START|>', '<|SPEECH_UNDERSTANDING_END|>']
    base_num = tokenizer.convert_tokens_to_ids(special_tokens[-1]) + 1
    tokenizer.pad_token = tokenizer.eos_token
    # transformers 5 removed encode_plus, which for one text was __call__.
    tokenizer.encode_plus = lambda text, **kwargs: tokenizer(text, **kwargs)
    training = []
    with tempfile.TemporaryDirectory() as root:
        cases = [("Hello there, how are you?", 9), ("A longer transcript; with punctuation!", 25)]
        transcriptions = {f"utt{i}": text for i, (text, _) in enumerate(cases)}
        for i, (_, frames) in enumerate(cases):
            np.save(os.path.join(root, f"utt{i}.wav.npy"), np.array([rng.randrange(65536) for _ in range(frames)]))
        namespace["init_worker"](transcriptions, root, tokenizer, 2048, base_num)
        dataset = namespace["TTSDataset"].__new__(namespace["TTSDataset"])
        rows = [namespace["process_audio_id"](f"utt{i}") for i in range(len(cases))]
        # TTSDataset.__init__ without the memmap files: the token ids it looks up, then the rows themselves.
        dataset.input_ids = np.stack(rows)
        dataset.length = len(rows)
        dataset.pad_token_id = tokenizer.pad_token_id
        dataset.tokenizer = tokenizer
        for name in ["speech_generation_start", "speech_generation_end", "text_generation_start", "text_generation_end",
                     "text_understanding_start", "text_understanding_end", "speech_understanding_start", "speech_understanding_end"]:
            setattr(dataset, f"{name}_id", tokenizer.convert_tokens_to_ids("<|" + name.upper() + "|>"))
        dataset.max_length = 2048
        dataset.ignore_index = -100
        for i, (text, _) in enumerate(cases):
            item = dataset[i]
            ids = [int(v) for v in item["input_ids"]]
            labels = [int(v) for v in item["labels"]]
            mask = [int(v) for v in item["attention_mask"]]
            # The pad token is <|eot_id|>, so the mask is 0 at every <|eot_id|> in the chat as well as in the padding: the
            # row ends at the template's <|eot_id|> after the end of speech.
            length = ids.index(dataset.speech_generation_end_id) + 2
            assert all(v == tokenizer.pad_token_id for v in ids[length - 1:])
            codes = np.load(os.path.join(root, f"utt{i}.wav.npy")).tolist()
            training.append({"text": text, "codes": codes, "ids": ids[:length], "labels": labels[:length], "mask": mask[:length]})

    stream = []
    for _ in range(60):
        roll = rng.random()
        if roll < 0.1:
            stream.append(rng.choice([128009, 128001, 128260, 128261, 1234, 15339]))
        else:
            stream.append(base_num + rng.randrange(65536))
    # The card's batch_decode over a 1-D tensor of ids decoded each id on its own (transformers 4.x); transformers 5 would
    # read a flat list as one sequence, so decode them one by one.
    codes = extract_speech_ids([tokenizer.decode([token], skip_special_tokens=True) for token in stream])

    result = {"date": DATE, "encodings": encodings, "prompts": prompts, "training": training,
              "stream": stream, "codes": codes, "base": base_num}
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "llasa_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False)
        handle.write("\n")
    print("base", base_num, "prompts", [len(p["ids"]) for p in prompts], "training", [len(t["ids"]) for t in training],
          "codes", len(codes), "of", len(stream))


if __name__ == "__main__":
    main()
