"""KaniTTS 2 through the release's own code (nineninesix-ai/kani-tts-2's kani_tts/model.py and core.py, and
kani-tts-2-pretrain's data collator and dataset processor), on a tiny model.

A tiny KaniTTS2ForCausalLM (LFM2 of width 16: conv, attention, conv, attention; learnable rotary scales with random
weights; a speaker projection from 6 values) over kanitts_reference.py's small byte-level tokenizer and a token layout
with four codebooks of 64 codes, so that a random model writes audio tokens and the frame-level positions matter.
Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/kanitts2_reference.json:
- the prompt with a language tag (KaniModel.get_input_ids) and a training sequence (TrainDataPreProcessor.create_input_ids,
  the tag lowercased);
- with a speaker embedding, the logits of the training sequence (inserted after its first token, frame-level positions
  shifted by one after it), the loss with the DataCollator's labels (the speaker slot unscored), and the gradients of the
  rotary scales and the speaker projection;
- greedy generation of 40 tokens with the speaker through generate() (prepare_inputs_for_generation's positions), at
  audio_step 1 and 0.5.
Float64, with the reference's float32 casts patched (RMSNorm, the rotary tables, the attention softmax, the learnable
rotary embedding). Pass --kani2 (kani-tts-2 checkout) and --pretrain (kani-tts-2-pretrain checkout). Default: verify;
--write: regenerate. Needs the .venv-kani environment and the committed kanitts_reference.json.
"""
import argparse
import importlib.util
import os
import sys
import types

import torch

import _fixture
from llama3_qwen3_reference import patch_float64

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/kanitts2_reference.json"
SOURCE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/kanitts_reference.json"
CODEBOOK = 64
SPEAKER_DIM = 6
TEXT = "Hello there, how are you?"
TAG = "EN_us"
ROPE_THETA = 1000000.0


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kani2", required=True)
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    from tokenizers import Tokenizer
    from transformers import Lfm2Config, PreTrainedTokenizerFast
    import transformers.models.lfm2.modeling_lfm2 as lfm2
    patch_float64(lfm2, "Lfm2")

    # The release's modules; NeMo's model class, datasets and the config loader are stubbed (nothing they compute is).
    for name in ("nemo", "nemo.collections", "nemo.collections.tts", "nemo.collections.tts.models"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["nemo.collections.tts.models"].AudioCodecModel = object
    sys.modules["datasets"] = types.ModuleType("datasets")
    sys.modules["datasets"].load_dataset = sys.modules["datasets"].concatenate_datasets = sys.modules["datasets"].load_from_disk = None
    sys.modules["datasets"].Dataset = object
    package = types.ModuleType("kani_tts")
    package.__path__ = [os.path.join(os.path.abspath(args.kani2), "kani_tts")]
    sys.modules["kani_tts"] = package
    model_module = load_file("kani_tts.model", os.path.join(os.path.abspath(args.kani2), "kani_tts/model.py"))
    core = load_file("kani_tts.core", os.path.join(os.path.abspath(args.kani2), "kani_tts/core.py"))
    data = load_file("kani_pretrain_data", os.path.join(os.path.abspath(args.pretrain), "utils/data.py"))
    # Training runs kani-tts-2-pretrain's own model, which inserts the speaker in forward() (the inference model inserts it
    # only while generating).
    train_module = load_file("kani_pretrain_model", os.path.join(os.path.abspath(args.pretrain), "utils/model.py"))
    processor_module = load_file("kani_pretrain_processor", os.path.join(os.path.abspath(args.pretrain), "dataset_processor.py"))

    # Float64 for the learnable rotary embedding, which computes in float32 like the standard one.
    def learnable_forward(self, x, position_ids):
        # The base frequencies too: the release computes inv_freq_base in float32.
        base = 1.0 / (ROPE_THETA ** (torch.arange(0, self.inv_freq_base.shape[0] * 2, 2, dtype=torch.float64) / (self.inv_freq_base.shape[0] * 2)))
        assert torch.allclose(base.float(), self.inv_freq_base.float(), rtol=1e-6)
        freqs = position_ids[:, :, None].double() * (base * self.alpha.double())[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)
    model_module.LearnableRotaryEmbedding.forward = learnable_forward
    train_module.LearnableRotaryEmbedding.forward = learnable_forward

    source = _fixture.load(SOURCE)
    stored = None if args.write else _fixture.load(FIXTURE)
    tok = Tokenizer.from_str(source["tokenizer"])
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok)
    text_vocab = source["text_vocabulary"]
    audio_start = text_vocab + 10
    vocab = audio_start + 4 * CODEBOOK
    config = dict(vocab_size=vocab, hidden_size=16, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                  layer_types=["conv", "full_attention", "conv", "full_attention"], conv_L_cache=3, conv_bias=False, norm_eps=1e-5,
                  rope_theta=ROPE_THETA, max_position_embeddings=256, block_auto_adjust_ff_dim=True, block_ffn_dim_multiplier=1.0,
                  block_multiple_of=8, tie_word_embeddings=True, pad_token_id=0, bos_token_id=1, eos_token_id=7)
    torch.set_default_dtype(torch.float64)

    def build(audio_step):
        torch.manual_seed(97)
        m = model_module.KaniTTS2ForCausalLM(Lfm2Config(**config, attn_implementation="eager"), audio_tokens_start=audio_start,
                                             tokens_per_frame=4, audio_step=audio_step, use_learnable_rope=True, speaker_emb_dim=SPEAKER_DIM)
        with torch.no_grad():
            for name, p in m.named_parameters():
                if name == "lm_head.weight":
                    continue
                if stored is not None:
                    p.copy_(torch.tensor(stored["state"][name], dtype=torch.float32).reshape(p.shape))
                elif name.endswith("norm.weight"):
                    p.copy_((1 + 0.1 * torch.randn_like(p)).float())
                elif name.endswith("alpha_weight"):
                    p.copy_((torch.randn_like(p)).float())
                else:
                    p.copy_((0.3 * torch.randn_like(p)).float())
        # from_pretrained gives the model a default GenerationConfig when the checkpoint has none.
        from transformers import GenerationConfig
        m.generation_config = GenerationConfig()
        return m.double()

    model = build(1.0)
    state32 = {k: v.float().clone() for k, v in model.named_parameters() if k != "lm_head.weight"}
    torch.manual_seed(101)
    speaker = (torch.randn(1, SPEAKER_DIM)).float().double() if args.write else torch.tensor([stored["speaker"]], dtype=torch.float64)

    # The prompt (core.KaniModel.get_input_ids) and a training sequence (dataset_processor.TrainDataPreProcessor).
    layout = dict(start_of_speech=text_vocab + 1, end_of_speech=text_vocab + 2, start_of_human=text_vocab + 3, end_of_human=text_vocab + 4,
                  start_of_ai=text_vocab + 5, end_of_ai=text_vocab + 6, pad_token=text_vocab + 7, audio_tokens_start=audio_start,
                  codebook_size=CODEBOOK, end_of_text=2, start_of_text=1, tokeniser_length=text_vocab)
    player = types.SimpleNamespace(**layout)
    kani = core.KaniModel.__new__(core.KaniModel)
    kani.player = player
    kani.tokenizer = hf_tokenizer
    prompt = kani.get_input_ids(TEXT, TAG)[0][0].tolist()
    processor = processor_module.TrainDataPreProcessor.__new__(processor_module.TrainDataPreProcessor)
    processor.text_tokenizer = hf_tokenizer
    processor.language_tag = TAG
    processor.__dict__.update(layout)
    g = torch.Generator().manual_seed(103)
    codes = torch.randint(0, CODEBOOK, (4, 6), generator=g).tolist()
    codes[0][3], codes[1][3], codes[2][3], codes[3][3] = codes[0][2], codes[1][2], codes[2][2], codes[3][2]
    example = processor.create_input_ids(processor.add_codes({"text": TEXT, **{f"nano_layer_{i + 1}": codes[i] for i in range(4)}}))
    training_ids = example["input_ids"]

    # Forward and loss as training runs them: the collator's labels, the training model's speaker insertion and positions.
    torch.manual_seed(97)
    trainer_model = train_module.FlashCompatibleLfm2ForCausalLM(Lfm2Config(**config, attn_implementation="eager"), audio_tokens_start=audio_start,
                                                                tokens_per_frame=4, audio_step=1.0, use_learnable_rope=True,
                                                                speaker_emb_dim=SPEAKER_DIM).double()
    trainer_model.load_state_dict(model.state_dict(), strict=True)
    model = trainer_model
    batch = data.DataCollator(pad_token_id=layout["pad_token"])([{"input_ids": training_ids, "speaker_emb": speaker[0].tolist()}])
    # The collator's attention mask goes in too, as the Trainer passes every batch key: without one, transformers 4.57's
    # create_causal_mask reads the repeated frame positions as packed sequences and blocks attention between them.
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], speaker_emb=batch["speaker_emb"].double())
    logits = out.logits
    labels = batch["labels"]
    # Precision patch: transformers' loss casts the logits to float32; the same shifted cross-entropy in float64.
    loss = torch.nn.functional.cross_entropy(logits[0, :-1], labels[0, 1:], ignore_index=-100)
    loss.backward()
    gradients = {name: p.grad.reshape(-1).tolist() for name, p in model.named_parameters()
                 if name.endswith("alpha_weight") or name == "model.speaker_emb_projection.weight"}

    generated = {}
    for step in (1.0, 0.5):
        m = build(step).eval()
        with torch.no_grad():
            out_ids = m.generate(torch.tensor([prompt]), attention_mask=torch.ones(1, len(prompt), dtype=torch.long), do_sample=False,
                                 max_new_tokens=40, repetition_penalty=1.1, eos_token_id=layout["end_of_speech"], pad_token_id=0,
                                 speaker_emb=speaker)
        generated[str(step)] = out_ids[0, len(prompt):].tolist()
    torch.set_default_dtype(torch.float32)

    result = {"config": config, "codebook": CODEBOOK, "speaker_dim": SPEAKER_DIM, "text": TEXT, "tag": TAG, "text_vocabulary": text_vocab,
              "speaker": speaker[0].tolist(), "prompt": prompt, "codes": codes, "training_ids": training_ids,
              "logits": logits.detach().reshape(-1).tolist(), "loss": float(loss.detach()), "gradients": gradients, "generated": generated}
    if args.write:
        result["state"] = {k: v.reshape(-1).tolist() for k, v in state32.items()}
        _fixture.save(FIXTURE, result, __file__)
        audio_tokens = sum(1 for t in generated["1.0"] if t >= audio_start)
        print("wrote", FIXTURE, "prompt", len(prompt), "training", len(training_ids), "loss", float(loss), "audio tokens", audio_tokens,
              "generated", generated)
        return
    check = _fixture.Comparison("KaniTTS 2")
    for key in ("prompt", "codes", "training_ids", "generated"):
        check.exact(key, stored[key], result[key])
    check.floats("logits", stored["logits"], result["logits"], 1e-12, float)
    check.floats("loss", [stored["loss"]], [result["loss"]], 1e-12, float)
    for name in stored["gradients"]:
        check.floats(name, stored["gradients"][name], gradients[name], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
