"""Original Spark-TTS Qwen/XLSR/BiCodec pretrained inference references.

Arguments: SparkAudio checkout, original pinned model directory, output JSON.
All neural calculations execute the original author/Hugging Face modules. Each
completed stage is cached with generator/source/file fingerprints for safe resume.
"""
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Model

checkout=Path(sys.argv[1]).resolve();weights=Path(sys.argv[2]).resolve();destination=Path(sys.argv[3]).resolve()
sys.path.insert(0,str(checkout))
torch.set_num_threads(1)
flat=lambda tensor:tensor.detach().flatten().tolist()
files=['LLM/model.safetensors','LLM/tokenizer.json','LLM/config.json','BiCodec/model.safetensors',
    'BiCodec/config.yaml','config.yaml','wav2vec2-large-xlsr-53/pytorch_model.bin',
    'wav2vec2-large-xlsr-53/config.json','wav2vec2-large-xlsr-53/preprocessor_config.json']
signature=dict(revision='642071559bfc6346c2359d19dcb6be3f9dd8a05d',
    code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip(),
    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    files={name:[(weights/name).stat().st_size,(weights/name).stat().st_mtime_ns] for name in files})
partial=Path(str(destination)+'.partial.json')
data=json.loads(partial.read_text()) if partial.exists() else dict(signature=signature)
if data.get('signature')!=signature:data=dict(signature=signature)
def save():
    destination.parent.mkdir(parents=True,exist_ok=True)
    partial.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')

if 'llm' not in data:
    from sparktts.utils.token_parser import TASK_TOKEN_MAP
    tokenizer=AutoTokenizer.from_pretrained(weights/'LLM',local_files_only=True)
    prompt=''.join([TASK_TOKEN_MAP['tts'],'<|start_content|>','Hello, world!','<|end_content|>',
        '<|start_global_token|>',''.join(f'<|bicodec_global_{i*13}|>' for i in range(32)),
        '<|end_global_token|>','<|start_semantic_token|>','<|bicodec_semantic_100|>'])
    texts=[prompt,'中文与 English 123','<|bicodec_global_4095|><|bicodec_semantic_8191|>']
    data['tokenizer_cases']=[dict(text=text,ids=tokenizer.encode(text,add_special_tokens=False)) for text in texts]
    ids=tokenizer(prompt,return_tensors='pt').input_ids
    model=AutoModelForCausalLM.from_pretrained(weights/'LLM',local_files_only=True,dtype=torch.float32).eval()
    data['llm']=dict(prompt=prompt,ids=flat(ids))
    for dtype,label in [(torch.float32,'float32'),(torch.float64,'float64')]:
        model.to(dtype)
        with torch.inference_mode():logits=model(ids,use_cache=False).logits[0,-1]
        data['llm'][label]=dict(logits=flat(logits),argmax=logits.argmax().item())
        print('Original released full Qwen complete:',label,flush=True)
    del model;gc.collect();save()

if 'xlsr' not in data:
    from sparktts.models.audio_tokenizer import BiCodecTokenizer
    torch.manual_seed(2295)
    time=torch.arange(16000,dtype=torch.float32)
    wave=.04*torch.randn(16000)+.18*torch.sin(time*.087)+.07*torch.cos(time*.034)
    processor=Wav2Vec2FeatureExtractor.from_pretrained(weights/'wav2vec2-large-xlsr-53',local_files_only=True)
    model=Wav2Vec2Model.from_pretrained(weights/'wav2vec2-large-xlsr-53',local_files_only=True).eval()
    model.config.output_hidden_states=True
    # Only provide the real loaded objects to the original feature-extraction method;
    # no BiCodec initialization is needed to execute this independent stage.
    helper=BiCodecTokenizer.__new__(BiCodecTokenizer);helper.processor=processor;helper.feature_extractor=model
    with torch.inference_mode():features=helper.extract_wav2vec2_features(wave.numpy())
    data['xlsr']=dict(audio=flat(wave),features=flat(features),frames=features.shape[1])
    print('Original released XLSR and original layers 11/14/16 mixing complete',flush=True)
    del helper,model;gc.collect();save()

if 'bicodec' not in data:
    from sparktts.models.bicodec import BiCodec
    from sparktts.models.audio_tokenizer import BiCodecTokenizer
    from sparktts.utils.file import load_config
    model=BiCodec.load_from_checkpoint(weights/'BiCodec')
    helper=BiCodecTokenizer.__new__(BiCodecTokenizer);helper.config=load_config(weights/'config.yaml')
    wave=torch.tensor(data['xlsr']['audio'],dtype=torch.float32)
    reference=torch.from_numpy(helper.get_ref_clip(wave.numpy()))[None,:]
    features=torch.tensor(data['xlsr']['features'],dtype=torch.float32).reshape(1,data['xlsr']['frames'],1024)
    data['bicodec']=dict(reference_samples=reference.shape[1])
    for dtype,label in [(torch.float32,'float32'),(torch.float64,'float64')]:
        model.to(dtype)
        with torch.inference_mode():
            semantic,global_tokens=model.tokenize(dict(feat=features.to(dtype),ref_wav=reference.to(dtype)))
            encoded=model.encoder(features.to(dtype).transpose(1,2))
            mel=model.mel_transformer(reference.to(dtype))
            _,speaker_features=model.speaker_encoder.speaker_encoder(mel.transpose(1,2),True)
            speaker_latents=model.speaker_encoder.perceiver_sampler(speaker_features.transpose(1,2))
            quantized=model.quantizer.detokenize(semantic)
            speaker=model.speaker_encoder.detokenize(global_tokens)
            prenet=model.prenet(quantized,speaker)
            postnet=model.postnet(prenet)
            waveform=model.detokenize(semantic,global_tokens)
        data['bicodec'][label]=dict(semantic=flat(semantic),global_tokens=flat(global_tokens),
            encoded=flat(encoded),mel=flat(mel),speaker=flat(speaker),prenet=flat(prenet),
            speaker_features=flat(speaker_features),speaker_latents=flat(speaker_latents),
            postnet=flat(postnet),waveform=flat(waveform))
        print('Original released full BiCodec tokenize/detokenize complete:',label,flush=True)
    del model;gc.collect();save()

if 'mel_float64_buffers' not in data:
    from sparktts.models.bicodec import BiCodec
    from sparktts.utils.file import load_config
    helper=BiCodec.__new__(BiCodec);torch.nn.Module.__init__(helper)
    # The untouched original construction, with the explicitly identified native precision.
    # Preserve the released float32-buffer and upcast-buffer results above independently.
    previous_dtype=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        helper.init_mel_transformer(load_config(weights/'BiCodec/config.yaml')['audio_tokenizer']['mel_params'])
    finally:torch.set_default_dtype(previous_dtype)
    samples=data['bicodec']['reference_samples']
    wave=torch.tensor(data['xlsr']['audio'],dtype=torch.float64)
    repeated=wave.repeat((samples+wave.numel()-1)//wave.numel())[:samples][None,:]
    with torch.inference_mode():matched_mel=helper.mel_transformer(repeated)
    data['mel_float64_buffers']=dict(mel=flat(matched_mel),source='Unchanged original init_mel_transformer, float64 default buffer construction')
    save()

destination.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print('Wrote complete Spark pretrained reference:',destination,flush=True)
