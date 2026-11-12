"""Complete released output policy after the independently generated real MBD waveform.

Original author normalization and neural forwards are unchanged. Explicit casts
retain the f32 WAV/libdf interfaces while matching native float64 neural math.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import typing as tp
import numpy as np
from scipy.io import wavfile
import torch
import torchaudio

parser = argparse.ArgumentParser()
for name in ['audiocraft-source', 'deepfilter-source', 'dependencies', 'ffmpeg-dependencies',
             'model-directory', 'decoder-reference', 'output']:
    parser.add_argument('--' + name, type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.dependencies)); sys.path.insert(0, str(args.ffmpeg_dependencies))
sys.path.insert(0, str(args.deepfilter_source / 'DeepFilterNet'))
import imageio_ffmpeg
from df.enhance import init_df, df_features
from df.io import resample
torch.set_num_threads(1)
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()


def run(command, **kwargs):
    assert command[0] == 'ffmpeg'
    return subprocess.run([ffmpeg] + command[1:], **kwargs)


scope = dict(torch=torch, torchaudio=torchaudio, tp=tp, sys=sys, Path=Path,
             sp=types.SimpleNamespace(run=run))
for path, names in [(args.audiocraft_source / 'audiocraft/data/audio_utils.py',
                     {'normalize_loudness', '_clip_wav', 'normalize_audio', 'f32_pcm'}),
                    (args.audiocraft_source / 'audiocraft/data/audio.py', {'_piping_to_ffmpeg'})]:
    nodes = [x for x in ast.parse(path.read_text(encoding='utf-8')).body
             if isinstance(x, ast.FunctionDef) and x.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
decoder = json.loads(args.decoder_reference.read_text(encoding='utf-8'))
raw = torch.tensor(decoder['waveform'], dtype=torch.float64).unsqueeze(0)
normalized = scope['normalize_audio'](raw, strategy='loudness', loudness_compressor=True, sample_rate=24000)
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'decoder.wav'
    scope['_piping_to_ffmpeg'](path, normalized.float(), 24000, ['-f', 'wav', '-c:a', 'pcm_s16le'])
    rate, pcm = wavfile.read(path)
assert rate == 24000 and pcm.dtype == np.int16
intermediate = torch.from_numpy(pcm.astype(np.float64) / 32768).unsqueeze(0)
resampled = resample(intermediate, 24000, 48000)
model, state, _ = init_df(str(args.model_directory))
model.double().eval()
with torch.no_grad():
    # The original libdf interface is float32 even with float64 neural weights.
    inputs = df_features(torch.nn.functional.pad(resampled.float(), (0, 960)), state, 96, device='cpu')
    spectrum = model(*(value.double() for value in inputs))[0].float()
    enhanced = torch.from_numpy(state.synthesis(torch.view_as_complex(spectrum.squeeze(1)).numpy().copy()))
    enhanced = enhanced[:, 480:resampled.shape[-1] + 480]
    final_pcm = (enhanced * 32768).to(torch.int16)
checkpoint = args.model_directory / 'checkpoints/model_120.ckpt.best'
result = dict(decoder_reference=str(args.decoder_reference), df_checkpoint=str(checkpoint),
              df_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              normalized=normalized.flatten().tolist(), decoder_pcm=pcm.tolist(),
              resampled=resampled.flatten().tolist(), enhanced=enhanced.flatten().tolist(),
              waveform=(final_pcm.double() / 32768).flatten().tolist(), sample_rate=48000,
              matched_neural_float64_with_original_float32_io=True)
args.output.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
print(json.dumps(dict(output=str(args.output), raw_samples=raw.numel(), output_samples=final_pcm.numel())))
