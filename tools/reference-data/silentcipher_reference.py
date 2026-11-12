"""Reference fixture for SilentCipher (Singh et al. 2024), the watermark Sesame's CSM applies: sony/silentcipher at d46d7d0
(server.Model.encode_wav and decode_wav, model.Encoder / CarrierDecoder / MsgDecoder) with the released 44.1 kHz
model's settings except a smaller transform and message decoder (n_fft 256, hop 128, message band 64, a 3-layer message
decoder of width 8) so that phase-shift decoding's 13 offsets run quickly; everything else follows hparams.yaml
(utterance-level normalization, negative messages, 21 symbols of 5 dims, VCTK's average energy).

Every parameter and buffer is a closed form of its name and flat index i: s = sin(0.37 i + 0.11 len(name) +
0.0013 (i^2 mod 9973)); batch-norm weights 1 + 0.2 s, running variances 1 + 0.5 s, running means and biases 0.05 s,
num_batches_tracked 0, and weights s * sqrt(2 / fan-in).

The release casts the phase to float32 (stft.py), builds the one-hot message as float32 and reads phase-shift search
audio as a FloatTensor; here those stay in float64.

Records the encoded audio (CSM's key [212, 211, 146, 56, 201] at 36 dB), the message decoder's scores on it, and
decode_wav's result with and without phase-shift decoding, for the audio and for the watermarked audio. With closed-form
weights the message decoder favours one symbol everywhere, so these decodes find no message; the released weights'
round trip is SilentCipherRealWeightsTests'.

Run with ~/source/repos/.venv-csm and the silentcipher clone (refsrc/csm-deps/silentcipher).
Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/silentcipher_reference.json.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "refsrc", "csm-deps", "silentcipher", "src"))
torch.set_default_dtype(torch.float64)

from silentcipher import server, stft  # noqa: E402
from silentcipher.model import CarrierDecoder, Encoder, MsgDecoder  # noqa: E402

FFT, HOP, BAND, MESSAGE_CHANNELS, MESSAGE_LAYERS = 256, 128, 64, 8, 3
KEY = [212, 211, 146, 56, 201]


def closed_form(name, shape):
    count = math.prod(shape) if shape else 1
    k = torch.arange(count, dtype=torch.int64)
    s = torch.sin(0.37 * k.to(torch.float64) + 0.11 * len(name) + 0.0013 * ((k * k) % 9973).to(torch.float64))
    if name.endswith("num_batches_tracked"):
        return torch.zeros(count)
    if name.endswith("bn.weight"):
        return 1 + 0.2 * s
    if name.endswith("running_var"):
        return 1 + 0.5 * s
    if len(shape) >= 2:
        return s * math.sqrt(2 * shape[0] / count)
    return 0.05 * s


def transform64(self, x):
    x = torch.nn.functional.pad(x, (0, self.win_len - x.shape[1] % self.win_len))
    fft = torch.stft(x, self.filter_length, self.hop_len, self.win_len, window=self.window.to(x), return_complex=True)
    real_part, imag_part = fft.real, fft.imag
    squared = real_part ** 2 + imag_part ** 2
    additive_epsilon = torch.ones_like(squared) * (squared == 0).double() * 1e-24
    magnitude = torch.sqrt(squared + additive_epsilon) - torch.sqrt(additive_epsilon)
    return magnitude, torch.atan2(imag_part, real_part)


def best_ps64(self, y_one_sec):
    def check_accuracy(pred_values):
        accuracy = 0
        for i in range(pred_values.shape[1]):
            unique, counts = np.unique(pred_values[:, i], return_counts=True)
            accuracy += np.max(counts) / pred_values.shape[0]
        return accuracy / pred_values.shape[1]
    y = torch.as_tensor(y_one_sec, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    max_accuracy, final_phase_shift = 0, 0
    for ps in range(0, self.config.HOP_LENGTH, 10):
        carrier, _ = self.stft.transform(y[0:1, 0:1, ps:].squeeze(1))
        msg_reconst = self.dec_m[0](carrier[:, None])
        pred_values = torch.argmax(msg_reconst[0, 0], dim=0).numpy()
        pred_values = pred_values[0:int(msg_reconst.shape[3] / self.config.message_len) * self.config.message_len]
        cur_acc = check_accuracy(pred_values.reshape([-1, self.config.message_len]))
        if cur_acc > max_accuracy:
            max_accuracy, final_phase_shift = cur_acc, ps
    return final_phase_shift


stft.STFT.transform = transform64
server.Model.get_best_ps = best_ps64


def build():
    config = argparse.Namespace(n_messages=1, model_type="n_msg", message_dim=5, message_len=21, enc_n_layers=3,
                                dec_c_n_layers=4, message_band_size=BAND, N_FFT=FFT, HOP_LENGTH=HOP, SR=44100,
                                ensure_negative_message=True, ensure_constrained_message=False, no_normalization=False,
                                frame_level_normalization=False, utterance_level_normalization=True, message_sdr=47)
    model = server.Model.__new__(server.Model)
    model.config, model.device = config, "cpu"
    model.n_messages, model.model_type, model.message_dim, model.message_len = 1, "n_msg", 5, 21
    model.enc_c = Encoder(n_layers=3, message_dim=5, out_dim=32, message_band_size=BAND, n_fft=FFT).eval()
    model.dec_c = CarrierDecoder(config=config, conv_dim=96, n_layers=4, message_band_size=BAND).eval()
    model.dec_m = [MsgDecoder(message_dim=5, message_band_size=BAND, channel_dim=MESSAGE_CHANNELS, num_layers=MESSAGE_LAYERS).eval()]
    model.average_energy_VCTK = 0.002837200844477648
    model.stft = stft.STFT(FFT, HOP)
    model.sr = 44100
    transform_message = model.enc_c.transform_message
    model.enc_c.transform_message = lambda msg: transform_message(msg.double())
    states = {}
    with torch.no_grad():
        for prefix, module in [("enc_c", model.enc_c), ("dec_c", model.dec_c), ("dec_m", model.dec_m[0])]:
            shapes = {}
            for name, value in module.state_dict().items():
                shapes[name] = list(value.shape)
                value.copy_(closed_form(name, list(value.shape)).reshape(value.shape).to(value.dtype))
            states[prefix] = shapes
    return model, states


def main():
    model, states = build()
    n = 44100 * 6 // 10
    i = torch.arange(n, dtype=torch.float64)
    generator = torch.Generator().manual_seed(0)
    audio = 0.3 * torch.sin(2 * math.pi * 440 * i / 44100) * (0.6 + 0.4 * torch.sin(2 * math.pi * 3 * i / 44100)) \
        + 0.05 * torch.randn(n, generator=generator)
    with torch.no_grad():
        encoded, _ = model.encode_wav(audio, 44100, KEY, message_sdr=36, calc_sdr=False)
        power = torch.mean(encoded ** 2)
        y = encoded * torch.sqrt(torch.tensor(model.average_energy_VCTK) / power)
        carrier, _ = model.stft.transform(y.view(1, -1))
        scores = model.dec_m[0](carrier[:, None])[0, 0]
        decodes = {}
        for label, signal in [("audio", audio), ("encoded", encoded)]:
            for shift in (False, True):
                decodes[f"{label}_{'shift' if shift else 'plain'}"] = model.decode_wav(signal, 44100, shift)
    result = {
        "fft": FFT, "hop": HOP, "band": BAND, "message_channels": MESSAGE_CHANNELS, "message_layers": MESSAGE_LAYERS,
        "shapes": states, "key": KEY, "audio": audio.tolist(), "encoded": encoded.tolist(),
        "scores": scores.tolist(), "decodes": decodes,
    }
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "silentcipher_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, default=float)
        handle.write("\n")
    print(json.dumps(decodes, default=float), float((encoded - audio).abs().max()))


if __name__ == "__main__":
    main()
