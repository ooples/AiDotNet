"""Separate original XLSR float64 arithmetic diagnostic, preserving float32 release results.

Arguments: original pinned Spark weight directory, generated Spark reference JSON.
Uses the original HF normalization method and unmodified Wav2Vec2Model. The normal
processor coerces inputs to float32; this explicitly labeled variant invokes its
unchanged normalization method on float64 values to match native double arithmetic.
"""
import json
from pathlib import Path
import sys
import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

weights=Path(sys.argv[1]);destination=Path(sys.argv[2]);data=json.loads(destination.read_text(encoding='utf-8'))
torch.set_num_threads(1)
processor=Wav2Vec2FeatureExtractor.from_pretrained(weights/'wav2vec2-large-xlsr-53',local_files_only=True)
wave=np.asarray(data['xlsr']['audio'],dtype=np.float64)
normalized=processor.zero_mean_unit_var_norm([wave],attention_mask=None,padding_value=0)[0]
model=Wav2Vec2Model.from_pretrained(weights/'wav2vec2-large-xlsr-53',local_files_only=True).double().eval()
with torch.inference_mode():
    output=model(torch.from_numpy(normalized)[None,:],output_hidden_states=True)
    mixed=(output.hidden_states[11]+output.hidden_states[14]+output.hidden_states[16])/3
data['xlsr_float64']=dict(source='Original HF normalization method and complete original XLSR backbone, float64 arithmetic',
    normalized=normalized.tolist(),features=mixed.flatten().tolist())
destination.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print('Added independent complete XLSR float64 precision comparison',flush=True)
