"""Original CosyVoice3 prompt methods using the released frozen ONNX checkpoints.

Arguments: author checkout, pinned weight directory, output JSON. Use the author's
requirements: openai-whisper 20231117, ONNX Runtime 1.18.0 and NumPy 1.26.4.
"""
import ast
import importlib.util
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Generator
from transformers import AutoTokenizer
import numpy as np
import onnxruntime
import soundfile
import torch
import torchaudio
import whisper
import torchaudio.compliance.kaldi as kaldi

checkout=Path(sys.argv[1]).resolve();weights=Path(sys.argv[2]).resolve();destination=Path(sys.argv[3]).resolve()
sys.path.insert(0,str(checkout));torch.set_num_threads(1)
from cosyvoice.utils.file_utils import load_wav
source=checkout/'cosyvoice/cli/frontend.py'
node=next(item for item in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(item,ast.ClassDef) and item.name=='CosyVoiceFrontEnd')
node.body=[item for item in node.body if isinstance(item,ast.FunctionDef) and item.name in ['_extract_speech_token','_extract_spk_embedding','_extract_text_token','_extract_speech_feat','frontend_zero_shot']]
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),globals())
helper=CosyVoiceFrontEnd();helper.device=torch.device('cpu')
tokenizer_source=checkout/'cosyvoice/tokenizer/tokenizer.py'
tokenizer_classes=[item for item in ast.parse(tokenizer_source.read_text(encoding='utf-8')).body
    if isinstance(item,ast.ClassDef) and item.name in ['CosyVoice2Tokenizer','CosyVoice3Tokenizer']]
exec(compile(ast.fix_missing_locations(ast.Module(body=tokenizer_classes,type_ignores=[])),str(tokenizer_source),'exec'),globals())
helper.tokenizer=CosyVoice3Tokenizer(weights/'CosyVoice-BlankEN');helper.allowed_special='all'
matcha=Path(sys.argv[4]).resolve() if len(sys.argv)>4 else checkout.parent/'Matcha-TTS'
audio_path=matcha/'matcha/utils/audio.py'
spec=importlib.util.spec_from_file_location('original_matcha_audio',audio_path)
matcha_audio=importlib.util.module_from_spec(spec);spec.loader.exec_module(matcha_audio)
helper.feat_extractor=partial(matcha_audio.mel_spectrogram,n_fft=1920,num_mels=80,sampling_rate=24000,
    hop_size=480,win_size=1920,fmin=0,fmax=None,center=False)
options=onnxruntime.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
helper.speech_tokenizer_session=onnxruntime.InferenceSession(str(weights/'speech_tokenizer_v3.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
helper.campplus_session=onnxruntime.InferenceSession(str(weights/'campplus.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
torch.manual_seed(3306)
data=dict(revision='29e01c4e8d000f4bcd70751be16fa94bf3d85a18',
    code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip(),
    whisper_version=whisper.__version__,ort_version=onnxruntime.__version__,numpy_version=np.__version__,cases=[])
for rate,samples in [(16000,16000),(24000,24000),(16000,20000)]:
    time=torch.arange(samples,dtype=torch.float32)
    wave=.04*torch.randn(samples)+.18*torch.sin(time*(2*torch.pi*230/rate))+.07*torch.cos(time*(2*torch.pi*85/rate))
    with tempfile.TemporaryDirectory(prefix='cosy-original-frontend-') as directory:
        wav=Path(directory)/'input.wav';soundfile.write(wav,wave.numpy(),rate,subtype='FLOAT')
        tokens,_=helper._extract_speech_token(str(wav));speaker=helper._extract_spk_embedding(str(wav))
        voice=helper.frontend_zero_shot('Hello, research world!','Hello, research world!',str(wav),24000,'')
        resampled=load_wav(str(wav),16000)
        mel=whisper.log_mel_spectrogram(resampled,n_mels=128)
        fbank=kaldi.fbank(resampled,num_mel_bins=80,dither=0,sample_frequency=16000)
        fbank=fbank-fbank.mean(dim=0,keepdim=True)
        precise_wave=wave.double()[None,:]
        if rate!=16000:precise_wave=torchaudio.transforms.Resample(rate,16000,dtype=torch.float64)(precise_wave)
        precise_fbank=kaldi.fbank(precise_wave,num_mel_bins=80,dither=0,sample_frequency=16000)
        precise_fbank=precise_fbank-precise_fbank.mean(dim=0,keepdim=True)
        precise_speaker=helper.campplus_session.run(None,{helper.campplus_session.get_inputs()[0].name:
            precise_fbank.unsqueeze(0).numpy().astype(np.float32)})[0]
        previous_dtype=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            matched_fbank=kaldi.fbank(precise_wave,num_mel_bins=80,dither=0,sample_frequency=16000)
        finally:torch.set_default_dtype(previous_dtype)
        matched_fbank=matched_fbank-matched_fbank.mean(dim=0,keepdim=True)
        matched_speaker=helper.campplus_session.run(None,{helper.campplus_session.get_inputs()[0].name:
            matched_fbank.unsqueeze(0).numpy().astype(np.float32)})[0]
    flat=lambda value:value.detach().flatten().tolist()
    data['cases'].append(dict(rate=rate,audio=flat(wave),tokens=flat(tokens),speaker=flat(speaker),
        resampled=flat(resampled),whisper_mel=flat(mel),mel_frames=mel.shape[2],fbank=flat(fbank),fbank_frames=fbank.shape[0]))
    data['cases'][-1].update(resampled_float64=flat(precise_wave),fbank_float64=flat(precise_fbank),
        speaker_float64_frontend=precise_speaker.flatten().tolist(),fbank_float64_buffers=flat(matched_fbank),
        speaker_float64_buffers=matched_speaker.flatten().tolist(),voice_tokens=flat(voice['flow_prompt_speech_token']),
        voice_text_tokens=flat(voice['prompt_text']),voice_mel=flat(voice['prompt_speech_feat']),voice_mel_frames=voice['prompt_speech_feat'].shape[1])
    print('Original frozen CosyVoice3 frontend complete:',rate,samples,flush=True)
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print('Wrote original CosyVoice3 frozen frontend reference:',destination,flush=True)
