"""GLM-4-Voice's released decoder (THUDM/glm-4-voice-decoder: flow.pt, hift.pt, config.yaml's sizes) on real speech tokens.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/glm_voice_decoder_realweights_reference.json what the release
makes, in float64, of the 38 speech tokens glm_speech_tokenizer_realweights_reference.json holds (the released tokenizer
on 3 s of speech) with the release's zero speaker embedding: the offline mel (flow.inference, no prompt) and the
paper's streaming decode (flow_inference.stream_inference: token2wav over blocks of 10). The noise and SineGen's draws are
replaced as glm_voice_decoder_reference.py replaces them.

Usage (default: verify; --write: regenerate):
  python tools/reference-data/glm_voice_decoder_realweights_reference.py --glm4voice <GLM-4-Voice checkout> \\
    --matcha <Matcha-TTS at dd9105b3> --decoder <glm-4-voice-decoder dir> [--write]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402
import glm_voice_decoder_reference as tiny  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/glm_voice_decoder_realweights_reference.json"
TOKENS_FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/glm_speech_tokenizer_realweights_reference.json"
MEL = 80


def build(decoder_dir):
    from omegaconf import DictConfig
    from cosyvoice.flow.decoder import ConditionalDecoder
    from cosyvoice.flow.flow import MaskedDiffWithXvec
    from cosyvoice.flow.flow_matching import ConditionalCFM
    from cosyvoice.flow.length_regulator import InterpolateRegulator
    from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor
    from cosyvoice.hifigan.generator import HiFTGenerator
    from cosyvoice.transformer.encoder import BlockConformerEncoder
    # config.yaml's flow and hift.
    encoder = BlockConformerEncoder(input_size=512, output_size=512, attention_heads=8, linear_units=2048, num_blocks=6,
                                    dropout_rate=0.1, positional_dropout_rate=0.1, attention_dropout_rate=0.1,
                                    normalize_before=True, input_layer="linear", pos_enc_layer_type="rel_pos_espnet",
                                    selfattention_layer_type="block_rel_selfattn", block_size=10, use_cnn_module=False,
                                    macaron_style=False)
    estimator = ConditionalDecoder(in_channels=320, out_channels=80, channels=[256, 256], dropout=0.0, attention_head_dim=64,
                                   n_blocks=4, num_mid_blocks=12, num_heads=8, act_fn="gelu")
    cfm = ConditionalCFM(in_channels=240, n_spks=1, spk_emb_dim=80, estimator=estimator,
                         cfm_params=DictConfig({"sigma_min": 1e-06, "solver": "euler", "t_scheduler": "cosine",
                                                "training_cfg_rate": 0.2, "inference_cfg_rate": 0.7, "reg_loss_type": "l1"}))
    flow = MaskedDiffWithXvec(input_size=512, output_size=80, spk_embed_dim=192, output_type="mel", vocab_size=16384,
                              input_frame_rate=12.5, only_mask_loss=True, encoder=encoder,
                              length_regulator=InterpolateRegulator(channels=80, sampling_ratios=[1, 1, 1, 1]), decoder=cfm)
    hift = HiFTGenerator(in_channels=80, base_channels=512, nb_harmonics=8, sampling_rate=22050, nsf_alpha=0.1, nsf_sigma=0.003,
                         nsf_voiced_threshold=10, upsample_rates=[8, 8], upsample_kernel_sizes=[16, 16],
                         istft_params={"n_fft": 16, "hop_len": 4}, resblock_kernel_sizes=[3, 7, 11],
                         resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], source_resblock_kernel_sizes=[7, 11],
                         source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]], lrelu_slope=0.1, audio_limit=0.99,
                         f0_predictor=ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512))
    flow.load_state_dict(torch.load(os.path.join(decoder_dir, "flow.pt"), map_location="cpu", weights_only=True))
    hift.load_state_dict(torch.load(os.path.join(decoder_dir, "hift.pt"), map_location="cpu", weights_only=True))
    from scipy.signal import get_window
    flow, hift = flow.double().eval(), hift.double().eval()
    hift.stft_window = torch.from_numpy(get_window("hann", 16, fftbins=True))
    return flow, hift


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--glm4voice", required=True)
    parser.add_argument("--matcha", required=True)
    parser.add_argument("--decoder", required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    tiny.stub_imports(args.glm4voice, args.matcha)
    tokens = _fixture.load(TOKENS_FIXTURE)["tokens"]
    stored = None if args.write else _fixture.load(FIXTURE)
    if stored:
        sequence = torch.tensor(stored["sequence"], dtype=torch.float64)
    else:
        sequence = torch.randn(MEL * 300, generator=torch.Generator().manual_seed(11)).float().double()
    tiny.NOISE = sequence
    tiny.STREAM = tokens
    flow, hift = build(args.decoder)
    embedding = torch.zeros(1, 192)
    with torch.inference_mode():
        mel = flow.inference(token=torch.tensor([tokens]), token_len=torch.tensor([len(tokens)]),
                             prompt_token=torch.zeros(1, 0, dtype=torch.int64), prompt_token_len=torch.tensor([0]),
                             prompt_feat=torch.zeros(1, 0, MEL), prompt_feat_len=torch.tensor([0]), embedding=embedding).clone()
        speech = tiny.stream(flow, hift, args.glm4voice, [10]).clone()
    results = [("mel", mel), ("speech", speech)]
    if args.write:
        data = {"tokens": tokens, "sequence": [float(v) for v in sequence.tolist()]}
        data.update({k: [float(x) for x in v.reshape(-1).tolist()] for k, v in results})
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, "mel", tuple(mel.shape), "speech", tuple(speech.shape))
        return
    check = _fixture.Comparison("GLM-4-Voice released decoder")
    for k, v in results:
        check.floats(k, stored[k], v.reshape(-1).tolist(), 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
