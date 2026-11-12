"""MetaVoice's original AudioCraft loudness/PCM handoff and DF output policy."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import typing as tp
import warnings
import numpy as np
from scipy.io import wavfile
import torch
import torchaudio

parser = argparse.ArgumentParser()
parser.add_argument('--audiocraft-source', type=Path, required=True)
parser.add_argument('--deepfilter-source', type=Path, required=True)
parser.add_argument('--dependencies', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--torchaudio-source', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.dependencies))
import imageio_ffmpeg
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
torch.set_num_threads(1)


def run(command, **kwargs):
    assert command[0] == 'ffmpeg'
    return subprocess.run([ffmpeg] + command[1:], **kwargs)


scope = dict(torch=torch, torchaudio=torchaudio, tp=tp, sys=sys, Path=Path,
             sp=types.SimpleNamespace(run=run))
sources = [(args.audiocraft_source / 'audiocraft/data/audio_utils.py',
            {'normalize_loudness', '_clip_wav', 'normalize_audio', 'f32_pcm'}),
           (args.audiocraft_source / 'audiocraft/data/audio.py', {'_piping_to_ffmpeg'})]
for path, names in sources:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)

cases = []
for name, length, amplitude in [('speech', 12001, .13), ('clamped_biquads', 14417, 1.7),
                                ('low_energy_short', 1377, .0001), ('silence', 1377, 0)]:
    position = torch.arange(length, dtype=torch.float64)
    audio = (torch.sin(position * .071) + .3 * torch.cos(position * .019)) * amplitude
    normalized = scope['normalize_audio'](audio.unsqueeze(0), strategy='loudness',
                                          loudness_compressor=True, sample_rate=24000).squeeze(0)
    loudness = None if length < 9600 else torchaudio.functional.loudness(audio.unsqueeze(0), 24000).item()
    # Execute the original f32le -> pcm_s16le ffmpeg writer, preserving its input protocol.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'source.wav'
        scope['_piping_to_ffmpeg'](path, normalized.float().unsqueeze(0), 24000,
                                  ['-f', 'wav', '-c:a', 'pcm_s16le'])
        rate, pcm = wavfile.read(path)
        assert rate == 24000 and pcm.dtype == np.int16
    cases.append(dict(name=name, audio=audio.tolist(), normalized=normalized.tolist(),
                      loudness=loudness, first_pcm=pcm.tolist()))

# Original DF save_audio conversion, including truncation and int16 wraparound.
boundary = torch.tensor([-1.1, -1., -.5, -.000045, -.000015, 0, .000015, .000045, .5, .99999, 1., 1.1])
df_scope = dict(torch=torch, ndarray=np.ndarray, Tensor=torch.Tensor, Union=tp.Union,
                Optional=tp.Optional, os=__import__('os'), ta=types.SimpleNamespace(
                    save=lambda path, audio, sr: df_scope.update(saved=audio.clone())))
df_path = args.deepfilter_source / 'DeepFilterNet/df/io.py'
tree = ast.parse(df_path.read_text(encoding='utf-8'))
node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'save_audio')
exec(compile(ast.Module(body=[node], type_ignores=[]), str(df_path), 'exec'), df_scope)
df_scope['save_audio']('unused.wav', boundary, 48000)
resample_audio = torch.linspace(-.2, .3, 1377, dtype=torch.float64)
resample_scope = dict(torch=torch, math=math, warnings=warnings, Optional=tp.Optional,
                      Tensor=torch.Tensor, _CPU=torch.device('cpu'))
names = {'_get_sinc_resample_kernel', '_apply_sinc_resample_kernel', 'resample'}
nodes = [node for node in ast.parse(args.torchaudio_source.read_text(encoding='utf-8')).body
         if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(nodes) == 3
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(args.torchaudio_source), 'exec'), resample_scope)
resampled = resample_scope['resample'](resample_audio, 24000, 48000,
                                     resampling_method='sinc_interpolation', lowpass_filter_width=16)
result = dict(audiocraft_revision=subprocess.check_output(['git', '-C', str(args.audiocraft_source), 'rev-parse', 'HEAD'], text=True).strip(),
              deepfilter_revision=subprocess.check_output(['git', '-C', str(args.deepfilter_source), 'rev-parse', 'HEAD'], text=True).strip(),
              ffmpeg_sha256=hashlib.sha256(Path(ffmpeg).read_bytes()).hexdigest(), cases=cases,
              boundary_audio=boundary.flatten().tolist(), boundary_pcm=df_scope['saved'].flatten().tolist(),
              resample_audio=resample_audio.tolist(), resampled=resampled.tolist())
args.output.write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
print(json.dumps(dict(output=str(args.output), cases=len(cases), resampled=len(resampled))))
