"""Diagnose released S3Gen mel arithmetic without loading neural weights.

Arguments: author checkout, released S3Gen reference JSON. Executes the original
mel function, then controlled precision variants of its cached window/filterbank.
The variants are explicitly labeled and never substituted for the released output.
"""
import importlib.util
import json
from pathlib import Path
import sys
import librosa
import numpy as np
import torch

torch.set_num_threads(1)
checkout=Path(sys.argv[1]); fixture=Path(sys.argv[2])
spec=importlib.util.spec_from_file_location('original_s3_mel',checkout/'src/chatterbox/models/s3gen/utils/mel.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
reference=json.loads(fixture.read_text())
audio=torch.tensor(reference['audio'],dtype=torch.float32)[None,:]
native=np.asarray(json.loads(Path(str(fixture)+'.native-mel.json').read_text()))
def execute(wave):
    return module.mel_spectrogram(wave,1920,80,24000,480,1920,0,8000,center=False).transpose(1,2).flatten().numpy()
cases={'released_float32':execute(audio)}
module.mel_basis={key:value.double() for key,value in module.mel_basis.items()}
module.hann_window={key:value.double() for key,value in module.hann_window.items()}
cases['float64_with_original_float32_buffers']=execute(audio.double())
module.mel_basis={key:torch.from_numpy(librosa.filters.mel(sr=24000,n_fft=1920,n_mels=80,fmin=0,fmax=8000,dtype=np.float64)) for key in module.mel_basis}
module.hann_window={key:torch.hann_window(1920,dtype=torch.float64) for key in module.hann_window}
cases['float64_with_float64_buffers']=execute(audio.double())
for name,values in cases.items():
    print(name,'native maximum error',float(np.max(np.abs(native-values))),flush=True)
Path(str(fixture)+'.mel-precision.json').write_text(json.dumps({key:value.tolist() for key,value in cases.items()},separators=(',',':')))
