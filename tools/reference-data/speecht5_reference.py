"""SpeechT5's model and objectives from Microsoft's own code (microsoft/SpeechT5, fairseq 7ca8bc1), on tiny sizes.

Builds T5TransformerModel from the official modules (the shared encoder with relative position bias, the decoder, the
text encoder pre-net, the speech encoder pre-net (wav2vec 2.0's feature extractor, the convolutional and sinusoidal
positions), the speech decoder pre-net and post-net, the text decoder pre-net and post-net, the HuBERT post-net and the
cross-modal Gumbel quantizer) and records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/speecht5_reference.json,
in float64 on float32 weights, with dropout off and the random choices given:
- text to speech: the decoder's outputs before and after the post-net, the stop logits and the cross-attentions on a
  teacher-forced target, and the fine-tuning loss (L1 + BCE with pos_weight 5 + guided attention on 2 heads of every
  layer, sigma 0.4); generation with threshold 0.5 and the 20x length limit;
- speech pre-training: the masked HuBERT logits (cosine against label embeddings / 0.1) with a given mask, the feature
  penalty, the quantizer's perplexities with a given mixing permutation, the reconstruction loss, and the total;
- text pre-training: the infilling cross-entropy on a given noised input and the total with the codebook term.
The precision patches replace float32 tables (sinusoids) with float64 ones.

Usage (default: verify; --write: regenerate), in the speecht5-ref image (refsrc/speecht5-docker):
  docker run --rm -v <repo>:/work -v <SpeechT5>/SpeechT5/speecht5:/code/speecht5:ro -w /work speecht5-ref \\
    python tools/reference-data/speecht5_reference.py [--write]
"""
import argparse
import math
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/code")
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/speecht5_reference.json"
D, FFN, HEADS, ENC_LAYERS, DEC_LAYERS, MELS, SPK = 16, 32, 2, 2, 2, 10, 6
TEXT_SYMBOLS = list("abcdefghij")
HUBERT_CLASSES = 12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args_cli = parser.parse_args()

    # Precision patches inside SpeechT5's own modules (their source casts to float32): the decoder's returned attention,
    # the HuBERT cosine, the feature penalty; and fairseq's GELU. The packages are registered as plain namespaces so their
    # __init__ (which imports every module) does not load the unpatched files first.
    import importlib
    import fairseq.modules
    import torch.nn.functional as F
    fairseq.modules.gelu = lambda x: F.gelu(x)
    quantizer_path = "/fairseq/fairseq/modules/gumbel_vector_quantizer.py"
    quantizer_source = open(quantizer_path, encoding="utf-8").read().replace(".float()", "")
    quantizer_module = types.ModuleType("fairseq.modules.gumbel_vector_quantizer")
    exec(compile(quantizer_source, quantizer_path, "exec"), quantizer_module.__dict__)
    fairseq.modules.GumbelVectorQuantizer = quantizer_module.GumbelVectorQuantizer
    for package in ("speecht5", "speecht5.models", "speecht5.models.modules", "speecht5.criterions", "speecht5.data", "speecht5.tasks"):
        module = types.ModuleType(package)
        module.__path__ = [os.path.join("/code", *package.split("."))]
        sys.modules[package] = module
    patches = {
        "speecht5.models.modules.decoder": [("attn = layer_attn.float().to(x)", "attn = layer_attn")],
        "speecht5.models.modules.speech_encoder_postnet": [("x.float(), targets.float(), dim=-1", "x, targets, dim=-1")],
        "speecht5.models.modules.speech_encoder_prenet": [("features_pen = x.float().pow(2).mean()", "features_pen = x.pow(2).mean()")],
        "speecht5.models.speecht5": [("logits_list = [x.float() for x in logits_list if x is not None]", "logits_list = [x for x in logits_list if x is not None]")],
        "speecht5.criterions.speech_pretrain_criterion": [("p = coef * p.float() * sample_size", "p = coef * p * sample_size")],
        "speecht5.criterions.text_pretrain_criterion": [("p = coef * p.float() * sample_size", "p = coef * p * sample_size")],
    }
    for name, replacements in patches.items():
        path = os.path.join("/code", *name.split(".")) + ".py"
        source = open(path, encoding="utf-8").read()
        for old_text, new_text in replacements:
            assert old_text in source, (name, old_text)
            source = source.replace(old_text, new_text)
        module = types.ModuleType(name)
        module.__file__ = path
        module.__package__ = name.rsplit(".", 1)[0]
        sys.modules[name] = module
        exec(compile(source, path, "exec"), module.__dict__)

    import fairseq.modules.sinusoidal_positional_embedding as sinusoid
    from fairseq.data import Dictionary
    import espnet.nets.pytorch_backend.transformer.embedding as espnet_embedding
    from speecht5.models.speecht5 import T5TransformerModel, base_architecture
    import speecht5.models.modules.speech_encoder_prenet as sep
    import speecht5.models.speecht5 as speecht5_model
    from speecht5.criterions.text_to_speech_loss import TexttoSpeechLoss
    from speecht5.criterions.speech_pretrain_criterion import SpeechPretrainCriterion
    from speecht5.criterions.text_pretrain_criterion import TextPretrainCriterion

    # Precision patches: the sinusoid tables are built in float32 whatever the default dtype.
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float64) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float64).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb
    sinusoid.SinusoidalPositionalEmbedding.get_embedding = staticmethod(get_embedding)

    def extend_pe(self, x):
        if self.pe is not None and self.pe.size(1) >= x.size(1):
            self.pe = self.pe.to(dtype=x.dtype)
            return
        position = torch.arange(0, x.size(1), dtype=torch.float64).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float64) * -(math.log(10000.0) / self.d_model))
        pe = torch.zeros(x.size(1), self.d_model, dtype=torch.float64)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).to(dtype=x.dtype)
    espnet_embedding.PositionalEncoding.extend_pe = extend_pe
    # fairseq's softmax and Fp32GroupNorm compute in float32; the same in float64.
    import torch.nn.functional as F
    import fairseq.utils as fairseq_utils
    import fairseq.modules.fp32_group_norm as fp32_group_norm
    fairseq_utils.softmax = lambda x, dim, onnx_trace=False: F.softmax(x, dim=dim)
    fairseq_utils.log_softmax = lambda x, dim, onnx_trace=False: F.log_softmax(x, dim=dim)
    fp32_group_norm.Fp32GroupNorm.forward = lambda self, input: F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)

    stored = None if args_cli.write else _fixture.load(FIXTURE)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2110)
    np.random.seed(2110)

    text_dict = Dictionary()
    for s in TEXT_SYMBOLS:
        text_dict.add_symbol(s)
    mask_idx = text_dict.add_symbol("<mask>")
    hubert_dict = Dictionary()
    for i in range(HUBERT_CLASSES):
        hubert_dict.add_symbol(str(i))

    a = argparse.Namespace(
        encoder_embed_dim=D, encoder_ffn_embed_dim=FFN, encoder_layers=ENC_LAYERS, encoder_attention_heads=HEADS,
        decoder_embed_dim=D, decoder_ffn_embed_dim=FFN, decoder_layers=DEC_LAYERS, decoder_attention_heads=HEADS,
        dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, encoder_layerdrop=0.0, decoder_layerdrop=0.0,
        use_conv_pos=True, use_sinc_pos=True, relative_position_embedding=True, encoder_max_relative_position=4,
        decoder_max_relative_position=4, layer_norm_first=False, encoder_normalize_before=False, decoder_normalize_before=False,
        max_text_positions=40, max_speech_positions=200, dprenet_units=8, dprenet_layers=2, dprenet_dropout_rate=0.0,
        postnet_chans=8, postnet_layers=5, postnet_filts=5, postnet_dropout_rate=0.0, spk_embed_dim=SPK, reduction_factor=2,
        speech_odim=MELS, conv_feature_layers="[(8,10,5)] + [(8,3,2)] * 2", conv_pos=8, conv_pos_groups=2, final_dim=6,
        latent_vars=5, latent_groups=2, use_codebook=True, codebook_prob=0.5, mask_prob=0.0, label_rates=50, sample_rate=16000,
        share_input_output_embed=True, bert_init=False, feature_grad_mult=1.0, extractor_mode="default",
        transformer_enc_positional_dropout_rate=0.0, transformer_dec_positional_dropout_rate=0.0, freeze_encoder_updates=0,
        freeze_decoder_updates=0, no_freeze_encoder_layer=None, use_sent_enc_layer=True, activation_fn="gelu",
    )
    base_architecture(a)
    embed = T5TransformerModel.build_embedding if hasattr(T5TransformerModel, "build_embedding") else None
    from fairseq.models.transformer import Embedding
    shared = Embedding(len(text_dict), D, text_dict.pad())
    model = T5TransformerModel(
        a,
        T5TransformerModel.build_encoder(a, text_dict, shared), T5TransformerModel.build_decoder(a),
        T5TransformerModel.build_text_encoder_prenet(shared, a), T5TransformerModel.build_speech_encoder_prenet(a),
        T5TransformerModel.build_text_decoder_prenet(shared, a), T5TransformerModel.build_speech_decoder_prenet(MELS, a),
        T5TransformerModel.build_text_decoder_postnet(shared, text_dict, a), T5TransformerModel.build_speech_decoder_postnet(MELS, a),
        None, T5TransformerModel.build_speech_encoder_postnet([hubert_dict], a),
    )
    model = model.double()   # some parameters are built as explicit float32 tensors
    with torch.no_grad():
        for name, p in model.named_parameters():
            if stored is not None:
                p.copy_(torch.tensor(stored["state"][name], dtype=torch.float32).reshape(p.shape))
            elif name.endswith("alpha"):
                p.copy_(torch.tensor(0.8).float())
            elif "norm" in name and name.endswith("weight"):
                p.copy_((1 + 0.2 * torch.randn_like(p)).float())
            elif name == "speech_decoder_postnet.prob_out.bias":
                p.copy_(torch.full_like(p, -1.0))   # so generation runs for several steps
            else:
                p.copy_((0.3 * torch.randn_like(p)).float())
        for name, b in model.named_buffers():
            if "running_mean" in name:
                b.copy_(torch.tensor(stored["buffers"][name]).reshape(b.shape) if stored else (0.1 * torch.randn_like(b)).float())
            elif "running_var" in name:
                b.copy_(torch.tensor(stored["buffers"][name]).reshape(b.shape) if stored else (1 + 0.2 * torch.rand_like(b)).float())
    model.eval()

    def tensor(key, shape, std=1.0):
        if stored is not None:
            return torch.tensor(stored[key]).reshape(shape)
        return (std * torch.randn(shape)).float().double()

    inputs = {
        "spk": tensor("spk", (1, SPK)), "mel": tensor("mel", (1, 13, MELS)), "wave": tensor("wave", (1, 1600), 0.3),
        "pt_mel": tensor("pt_mel", (1, 10, MELS)),
    }
    tokens = torch.tensor([[4, 7, 5, 9, 11, 6, text_dict.eos()]])
    out = {}
    with torch.no_grad():
        # ---- text to speech (t2s), teacher forced
        mel = inputs["mel"]
        frames = mel.size(1)
        mel_in = mel[:, 1::2]
        prev = torch.cat([mel_in.new_zeros(1, 1, MELS), mel_in[:, :-1]], dim=1)
        model.use_codebook = False   # fine-tuned TTS does not quantize
        before, after, logits, attn = model(src_tokens=tokens, prev_output_tokens=prev, tgt_lengths=torch.tensor([mel_in.size(1)]),
                                            spkembs=inputs["spk"], task_name="t2s")
        out.update(before=before[0], after=after[0], logits=logits[0], attn=torch.stack([x[0] for x in attn]))
        labels = mel.new_zeros(1, frames)
        labels[0, frames - 1:] = 1.0
        crit = TexttoSpeechLoss(task=None, sentence_avg=True, use_guided_attn_loss=True)
        sample = {"labels": labels, "dec_target": mel, "dec_target_lengths": torch.tensor([frames]), "src_lengths": torch.tensor([tokens.size(1)])}
        loss, l1, l2, bce, attn_loss = crit.compute_loss(model, (before, after, logits, attn), sample)
        out.update(tts_loss=loss.reshape(1), tts_l1=l1.reshape(1), tts_bce=bce.reshape(1), tts_attn=attn_loss.reshape(1))

        # ---- generation
        outs, probs, _ = model.generate_speech(src_tokens=tokens, spkembs=inputs["spk"])
        out.update(generated=outs, generated_probs=probs)

        # ---- speech pre-training, with the mask and the codebook mixing given
        model.use_codebook = True
        wave = inputs["wave"]
        feat_frames = 15   # (1600 - 10) // 5 + 1 = 319 -> 159 -> 79... set below from the extractor
        x = model.speech_encoder_prenet.feature_extractor(wave)
        feat_frames = x.size(2)
        mask = np.zeros((1, feat_frames), dtype=bool)
        mask[0, 3:9] = True
        mask[0, 20:24] = True
        sep.compute_mask_indices = lambda *args, **kwargs: mask
        model.speech_encoder_prenet.mask_prob = 0.5
        permutation = torch.tensor([5, 0, 12, 3, 7, 9, 1])
        torch_randperm = torch.randperm
        torch.randperm = lambda n, *args, **kwargs: torch.cat([permutation[permutation < n], torch.tensor([i for i in range(n) if i not in permutation.tolist()], dtype=torch.long)])
        hubert = torch.tensor([[(3 * t + 1) % HUBERT_CLASSES for t in range(feat_frames)]])
        pt_mel = inputs["pt_mel"]
        pt_in = pt_mel[:, 1::2]
        pt_prev = torch.cat([pt_in.new_zeros(1, 1, MELS), pt_in[:, :-1]], dim=1)
        pt_labels = pt_mel.new_zeros(1, pt_mel.size(1))
        pt_labels[0, -1] = 1.0
        speech_sample = {
            "id": torch.tensor([0]), "target_list": [hubert],
            "net_input": {"source": wave, "padding_mask": torch.zeros(wave.shape, dtype=torch.bool), "prev_output_tokens": pt_prev,
                          "spkembs": inputs["spk"], "tgt_lengths": torch.tensor([pt_in.size(1)]), "task_name": "speech_pretrain"},
            "labels": pt_labels, "dec_target": pt_mel, "dec_target_lengths": torch.tensor([pt_mel.size(1)]),
            "src_lengths": torch.tensor([wave.size(1)]), "task_name": "speech_pretrain",
        }
        speech_crit = SpeechPretrainCriterion(task=None, sentence_avg=True, pred_masked_weight=1.0, pred_nomask_weight=0.0, loss_weights=[10, 0.1])
        net_output, net_output_dec = model(target_list=speech_sample["target_list"], **speech_sample["net_input"])
        out["hubert_logits"] = net_output["logit_m_list"][0]
        out["features_pen"] = net_output["features_pen"].reshape(1)
        out["prob_perplexity"] = net_output["prob_perplexity"].reshape(1)
        out["pt_after"] = net_output_dec[1][0]
        speech_loss, sample_size, log = speech_crit(model, speech_sample)
        out["speech_loss"] = speech_loss.reshape(1)
        out["speech_sample_size"] = torch.tensor([float(sample_size)])

        # ---- text pre-training (infilling), with the codebook mixing given
        noised = torch.tensor([[4, mask_idx, 9, mask_idx, 6, text_dict.eos()]])
        target = torch.tensor([[4, 7, 5, 9, 11, 10, 6, text_dict.eos()]])
        prev_text = torch.cat([torch.tensor([[text_dict.eos()]]), target[:, :-1]], dim=1)
        text_sample = {"net_input": {"src_tokens": noised, "src_lengths": torch.tensor([noised.size(1)]), "prev_output_tokens": prev_text,
                                     "task_name": "text_pretrain"},
                       "target": target, "ntokens": target.numel(), "task_name": "text_pretrain"}
        text_crit = TextPretrainCriterion(task=types.SimpleNamespace(dicts={"text": text_dict}, target_dictionary=text_dict), sentence_avg=False, bart_weight=1.0, loss_weights=[10, 0.1])
        (text_logits, _), codebook_out, _ = model(**text_sample["net_input"])
        out["text_logits"] = text_logits[0]
        text_loss, _, _ = text_crit(model, text_sample)
        out["text_loss"] = text_loss.reshape(1)
        torch.randperm = torch_randperm

    if args_cli.write:
        data = {"state": {n: _fixture.flat(p) for n, p in model.named_parameters()},
                "buffers": {n: _fixture.flat(b) for n, b in model.named_buffers() if "running" in n},
                "tokens": tokens[0].tolist(), "noised": noised[0].tolist(), "target": target[0].tolist(), "mask": mask[0].tolist(),
                "hubert": hubert[0].tolist(), "permutation": permutation.tolist(), "mask_idx": mask_idx, "eos": text_dict.eos(),
                "pad": text_dict.pad(), "text_vocab": len(text_dict), "sample_size": sample_size}
        data.update({k: _fixture.flat(v) for k, v in inputs.items()})
        data.update({k: [float(x) for x in v.reshape(-1).tolist()] for k, v in out.items()})
        data["shapes"] = {k: list(v.shape) for k, v in out.items()}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, {k: list(v.shape) for k, v in out.items()})
        return
    check = _fixture.Comparison("SpeechT5")
    for k, v in out.items():
        check.floats(k, stored[k], [float(x) for x in v.reshape(-1).tolist()], dtype="float64")
    check.report()


if __name__ == "__main__":
    main()
