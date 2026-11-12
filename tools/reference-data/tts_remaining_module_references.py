"""Reference tensors from the actual released UniAudio, XTTS and VoiceCraft modules.

Arguments: UniAudio/model.py, coqui-TTS checkout, VoiceCraft checkout.
Requires PyTorch, transformers, einops, numpy, torchmetrics and huggingface_hub.
The original modules execute the neural computations; no native implementation is mirrored here.
"""
import argparse
import importlib.util
import json
import pathlib
import sys
import types

import torch


ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
SHAPES = {}


def weights(module, prefix=''):
    SHAPES.update({prefix + k: list(v.shape) for k, v in module.state_dict().items()})
    return {prefix + k: v.detach().flatten().tolist() for k, v in module.state_dict().items()}


def write(name, data):
    data['weight_shapes'] = dict(SHAPES)
    SHAPES.clear()
    (OUT / name).write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
    print('Wrote', name)


def uniaudio(path):
    spec = importlib.util.spec_from_file_location('official_uniaudio', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.manual_seed(2294)
    model = module.MegaByteModel(n_vocab=64, n_ctx=32, n_state=12, n_head=3, n_layer=2,
                                prefix_lm=True, na_repeat=3, local_sos=1, global_sos=2).eval()
    sequence = torch.tensor([[3, 3, 3, 4, 4, 4, 10, 18, 26, 11, 19, 27]])
    mask = torch.tensor([[False] * 6 + [True] * 6])
    with torch.no_grad():
        logits = model(sequence, mask)
    write('uniaudio_lm_reference.json', dict(source='yangdongchao/UniAudio/UniAudio/model.py',
        weights=weights(model), sequence=sequence.flatten().tolist(), prefix_tokens=6, logits=logits.flatten().tolist()))


def xtts(checkout):
    # Only bypass package __init__ files that eagerly import unrelated training losses.
    # Each referenced neural module is still imported unchanged from the upstream checkout.
    package = pathlib.Path(checkout) / 'TTS'
    for name, directory in [('TTS', package), ('TTS.tts', package / 'tts'),
                            ('TTS.tts.layers', package / 'tts/layers'), ('TTS.tts.layers.xtts', package / 'tts/layers/xtts')]:
        mod = types.ModuleType(name)
        mod.__path__ = [str(directory)]
        sys.modules[name] = mod
    from TTS.tts.layers.xtts.gpt import GPT
    from TTS.tts.layers.xtts.latent_encoder import ConditioningEncoder
    from TTS.tts.layers.xtts.perceiver_encoder import PerceiverResampler
    torch.manual_seed(2294)
    model = GPT(start_text_token=1, stop_text_token=2, layers=1, model_dim=32, heads=4,
        max_text_tokens=64, max_mel_tokens=2, number_text_tokens=32, num_audio_tokens=18,
        start_audio_token=16, stop_audio_token=17, max_prompt_tokens=70, use_perceiver_resampler=False).double().eval()
    with torch.no_grad():
        model.gpt.ln_f.weight.uniform_(0.7, 1.3)
        model.gpt.ln_f.bias.normal_(0, .1)
        model.final_norm.weight.uniform_(0.7, 1.3)
        model.final_norm.bias.normal_(0, .1)
    text = torch.tensor([[1, 2, 3]])
    audio = torch.tensor([[16, 4, 5]])
    condition = torch.randn(1, 2, 32, dtype=torch.float64)
    with torch.no_grad():
        te = model.text_embedding(text) + model.text_pos_embedding(text)
        ae = model.mel_embedding(audio) + model.mel_pos_embedding(audio)
        text_hidden, audio_hidden = model.get_logits(te, model.text_head, ae, model.mel_head,
            prompt=condition, return_latent=True)
    encoder = ConditioningEncoder(80, 32, attn_blocks=1, num_attn_heads=4).eval()
    perceiver = PerceiverResampler(dim=32, depth=1, dim_context=32, num_latents=2, dim_head=64,
        heads=1, ff_mult=4, use_flash_attn=False).eval()
    with torch.no_grad():
        encoder.attn[0].proj_out.weight.normal_(0, .05)
        mel = torch.randn(1, 80, 5)
        conditioning = perceiver(encoder(mel).transpose(1, 2))
    state = weights(model, 'gpt.')
    state.update(weights(encoder, 'gpt.conditioning_encoder.'))
    state.update(weights(perceiver, 'gpt.conditioning_perceiver.'))
    # torch interpolate's two scale_factor passes are intentionally recorded separately.
    interpolation = torch.arange(6, dtype=torch.float64).view(1, 2, 3)
    first = torch.nn.functional.interpolate(interpolation, scale_factor=1024/256, mode='linear')
    second = torch.nn.functional.interpolate(first, scale_factor=24000/22050, mode='linear')
    write('xtts_gpt_reference.json', dict(source='coqui-ai/TTS/TTS/tts/layers/xtts/gpt.py', weights=state,
        text=text.flatten().tolist(), audio=audio.flatten().tolist(), condition=condition.flatten().tolist(),
        text_hidden=text_hidden.flatten().tolist(), audio_hidden=audio_hidden.flatten().tolist(),
        mel=mel.flatten().tolist(), conditioning=conditioning.flatten().tolist(),
        interpolation_input=interpolation.transpose(1, 2).flatten().tolist(),
        interpolation_output=second.transpose(1, 2).flatten().tolist(), interpolation_frames=second.shape[-1]))

    from TTS.tts.layers.xtts.dvae import DiscreteVAE
    torch.manual_seed(733)
    vae = DiscreteVAE(positional_dims=1, num_tokens=8, codebook_dim=4, num_layers=2,
        num_resnet_blocks=1, hidden_dim=4, channels=4, kernel_size=3,
        normalization=None, use_transposed_convs=False).double().eval()
    with torch.no_grad():
        input_mel = torch.randn(1, 4, 8, dtype=torch.float64)
        codes = vae.get_codebook_indices(input_mel)
        reconstructed, _ = vae.decode(codes)
    write('xtts_dvae_reference.json', dict(source='coqui-ai/TTS/TTS/tts/layers/xtts/dvae.py',
        weights=weights(vae), mel=input_mel.flatten().tolist(), codes=codes.flatten().tolist(),
        reconstructed=reconstructed.transpose(1, 2).flatten().tolist()))


def voicecraft(checkout):
    sys.path.insert(0, str(checkout))
    from models.voicecraft import VoiceCraft
    config = dict(n_codebooks=4, audio_vocab_size=8, n_special=4, empty_token=8, eog=9,
        audio_pad_token=10, eos=11, text_vocab_size=80, text_pad_token=80, d_model=16,
        audio_embedding_dim=16, text_embedding_dropout=0, audio_embedding_dropout=0,
        text_positional_embedding_dropout=0, audio_positional_embedding_dropout=0,
        trm_dropout=0, nhead=4, num_decoder_layers=1, max_n_spans=3)
    torch.manual_seed(2294)
    model = VoiceCraft(argparse.Namespace(**config)).double().eval()
    text = torch.tensor([[4, 5, 6]])
    rows = torch.tensor([[[1, 2, 3, 4], [9, 9, 9, 9], [2, 3, 4, 5]]])
    with torch.no_grad():
        x = model.text_positional_embedding(model.text_embedding(text))
        y = sum(model.audio_embedding[q](rows[:, :, q]) for q in range(4))
        y[:, 1] = model.mask_embedding[0]
        y = model.audio_positional_embedding(y)
        n, m = text.shape[1], rows.shape[1]
        xm = torch.ones(n, n, dtype=torch.bool).triu(1)
        ym = torch.ones(m, m, dtype=torch.bool).triu(1)
        result = model.dec_forward(x, torch.tensor([n]), xm, torch.zeros(1, n, dtype=torch.bool),
            y, torch.tensor([m]), ym, torch.zeros(1, m, dtype=torch.bool))[0]
        logits = [head(result).flatten().tolist() for head in model.predict_layer]
    write('voicecraft_lm_reference.json', dict(source='jasonppy/VoiceCraft/models/voicecraft.py',
        weights=weights(model), text=text.flatten().tolist(), rows=rows[0].tolist(), masks=[-1, 0, -1], logits=logits))


if __name__ == '__main__':
    uniaudio(sys.argv[1])
    xtts(sys.argv[2])
    voicecraft(sys.argv[3])
