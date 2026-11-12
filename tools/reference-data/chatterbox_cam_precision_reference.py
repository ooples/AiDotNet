"""Original released CAM++ weights with explicit float64 frontend/network arithmetic.

Arguments: author checkout, original weights, released S3Gen reference JSON.
This precision variant calls forward directly because original inference forcibly
casts features to float32. The unmodified float32 result remains in the main fixture.
"""
import importlib.util
import json
from pathlib import Path
import sys
import torch
import torchaudio
from safetensors import safe_open

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
checkout=Path(sys.argv[1]);weights=Path(sys.argv[2]);fixture=Path(sys.argv[3])
spec=importlib.util.spec_from_file_location('original_s3_xvector',checkout/'src/chatterbox/models/s3gen/xvector.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
model=module.CAMPPlus(memory_efficient=False).double().eval()
with safe_open(str(weights/'s3gen.safetensors'),framework='pt') as source:
    model.load_state_dict({key.removeprefix('speaker_encoder.'):source.get_tensor(key) for key in source.keys() if key.startswith('speaker_encoder.')},strict=True)
root=json.loads(fixture.read_text())
audio=torch.tensor(root['audio'],dtype=torch.float64)[None,:]
with torch.inference_mode():
    resampled=torchaudio.functional.resample(audio,24000,16000)
    features,_,_=module.extract_feature(resampled)
    embedding=model(features)
flat=lambda value:value.flatten().tolist()
output=dict(audio16=flat(resampled),features=flat(features),frames=features.shape[1],embedding=flat(embedding))
destination=Path(str(fixture)+'.cam-precision.json')
destination.write_text(json.dumps(output,separators=(',',':')))
print('Wrote original CAM++ precision variant:',destination)
