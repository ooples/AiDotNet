"""Original CosyVoice-300M / CosyVoice2 frozen frontend and zero-shot prompt.

Arguments: author checkout, original release directory, output JSON, variant (1/2),
Matcha-TTS checkout. Use the author's whisper 20231117, ORT 1.18 and NumPy 1.26.
Only optional constructor imports are excluded; original extraction methods run
unchanged with their actual tokenizer and original ONNX checkpoints.
"""
import ast, importlib.util, json, subprocess, sys, tempfile
from functools import partial
from pathlib import Path
from typing import Generator
import numpy as np
import onnxruntime
import soundfile
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper
from transformers import AutoTokenizer

checkout, release, destination = map(Path, sys.argv[1:4])
variant = int(sys.argv[4]); matcha = Path(sys.argv[5])
sys.path.insert(0, str(checkout.resolve())); torch.set_num_threads(1)
from cosyvoice.utils.file_utils import load_wav

source = checkout/'cosyvoice/cli/frontend.py'
node = next(item for item in ast.parse(source.read_text(encoding='utf-8')).body
            if isinstance(item, ast.ClassDef) and item.name == 'CosyVoiceFrontEnd')
node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name in
    ['_extract_speech_token', '_extract_spk_embedding', '_extract_text_token', '_extract_speech_feat', 'frontend_zero_shot']]
exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), 'exec'), globals())
helper = CosyVoiceFrontEnd(); helper.device = torch.device('cpu'); helper.allowed_special = 'all'
tensor_frontend = helper._extract_speech_token.__code__.co_varnames[1] == 'speech'
if variant == 1:
    # Exactly the original 300M YAML: whisper.tokenizer.get_tokenizer, 100 languages.
    from whisper.tokenizer import get_tokenizer
    helper.tokenizer = get_tokenizer(multilingual=True, num_languages=100, language='en', task='transcribe')
else:
    path = checkout/'cosyvoice/tokenizer/tokenizer.py'
    classes = [item for item in ast.parse(path.read_text(encoding='utf-8')).body
               if isinstance(item, ast.ClassDef) and item.name == 'CosyVoice2Tokenizer']
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])), str(path), 'exec'), globals())
    helper.tokenizer = CosyVoice2Tokenizer(release/'CosyVoice-BlankEN')
target_rate, fft, hop, maximum = (22050, 1024, 256, 8000) if variant == 1 else (24000, 1920, 480, None)
spec = importlib.util.spec_from_file_location('original_matcha_audio', matcha/'matcha/utils/audio.py')
matcha_audio = importlib.util.module_from_spec(spec); spec.loader.exec_module(matcha_audio)
helper.feat_extractor = partial(matcha_audio.mel_spectrogram, n_fft=fft, num_mels=80, sampling_rate=target_rate,
    hop_size=hop, win_size=fft, fmin=0, fmax=maximum, center=False)
options = onnxruntime.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
helper.speech_tokenizer_session = onnxruntime.InferenceSession(str(release/f'speech_tokenizer_v{variant}.onnx'),
    sess_options=options, providers=['CPUExecutionProvider'])
helper.campplus_session = onnxruntime.InferenceSession(str(release/'campplus.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
torch.manual_seed(variant*1000+336)
flat = lambda value: value.detach().flatten().tolist()
data = dict(variant=variant, target_rate=target_rate, tensor_frontend=tensor_frontend, code_revision=subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip(),
    whisper_version=whisper.__version__, ort_version=onnxruntime.__version__, numpy_version=np.__version__, cases=[])
cases = [(16000, 16000), (16000, 20000), (16000, 24000)] if tensor_frontend else \
    [(16000, 16000), (target_rate, target_rate), (16000, 20000)]
for rate, samples in cases:
    t = torch.arange(samples, dtype=torch.float32)
    wave = .04*torch.randn(samples)+.18*torch.sin(t*(2*torch.pi*230/rate))+.07*torch.cos(t*(2*torch.pi*85/rate))
    with tempfile.TemporaryDirectory(prefix='cosy-author-frontend-') as directory:
        path = Path(directory)/'input.wav'; soundfile.write(path, wave.numpy(), rate, subtype='FLOAT')
        prompt = load_wav(str(path), 16000) if tensor_frontend else str(path)
        tokens, _ = helper._extract_speech_token(prompt); speaker = helper._extract_spk_embedding(prompt)
        voice = helper.frontend_zero_shot('Hello, research world!', 'Hello, research world!', prompt, target_rate, '')
        audio = load_wav(str(path), 16000); mel = whisper.log_mel_spectrogram(audio, n_mels=128)
        features = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
        features = features-features.mean(dim=0, keepdim=True)
        precise = wave.double()[None]
        if rate != 16000: precise = torchaudio.transforms.Resample(rate, 16000, dtype=torch.float64)(precise)
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            matched = kaldi.fbank(precise, num_mel_bins=80, dither=0, sample_frequency=16000)
        finally: torch.set_default_dtype(previous)
        matched = matched-matched.mean(dim=0, keepdim=True)
        matched_speaker = helper.campplus_session.run(None, {helper.campplus_session.get_inputs()[0].name:
            matched.unsqueeze(0).numpy().astype(np.float32)})[0]
        precise_target = wave.double()[None]
        if rate != target_rate: precise_target = torchaudio.transforms.Resample(rate, target_rate, dtype=torch.float64)(precise_target)
        mel_key = str(maximum)+'_cpu'; window_key = 'cpu'
        old_mel = matcha_audio.mel_basis[mel_key]; old_window = matcha_audio.hann_window[window_key]
        try:
            matcha_audio.mel_basis[mel_key] = torch.from_numpy(matcha_audio.librosa_mel_fn(
                sr=target_rate, n_fft=fft, n_mels=80, fmin=0, fmax=maximum, dtype=np.float64))
            matcha_audio.hann_window[window_key] = torch.hann_window(fft, dtype=torch.float64)
            precise_mel = helper.feat_extractor(precise_target).squeeze(0).transpose(0, 1)
            if variant == 2: precise_mel = precise_mel[:2*voice['flow_prompt_speech_token'].shape[1]]
        finally:
            matcha_audio.mel_basis[mel_key] = old_mel; matcha_audio.hann_window[window_key] = old_window
    data['cases'].append(dict(rate=rate, audio=flat(wave), tokens=flat(tokens), speaker=flat(speaker),
        resampled=flat(audio), whisper_mel=flat(mel), mel_frames=mel.shape[2], fbank=flat(features), fbank_frames=features.shape[0],
        resampled_float64=flat(precise), fbank_float64_buffers=flat(matched), speaker_float64_buffers=matched_speaker.flatten().tolist(),
        voice_tokens=flat(voice['flow_prompt_speech_token']), voice_text_tokens=flat(voice['prompt_text']),
        voice_mel=flat(voice['prompt_speech_feat']), voice_mel_frames=voice['prompt_speech_feat'].shape[1],
        voice_mel_float64_buffers=flat(precise_mel)))
    print('Original frozen CosyVoice frontend:', variant, rate, samples, flush=True)
destination.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
