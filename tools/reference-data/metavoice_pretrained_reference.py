"""Execute the unchanged official MetaVoice GPTs with all released weights.

The 1.2B checkpoint stays memory mapped during import. Original stored-float32
and explicitly matched float64 results are both retained. No neural forward is
replaced, and causal input/output weight tying is restored after assign-style loading.
"""
import argparse
import base64
import gc
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--release', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))
from fam.llm.model import GPT, GPTConfig
from fam.quantiser.text.tokenise import TrainedBPETokeniser

torch.set_num_threads(1)
release = json.loads(args.release.read_text(encoding='utf-8'))
source_revision = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
result = dict(repository=release['repository'], revision=release['revision'], source_revision=source_revision,
    source_sha256=hashlib.sha256(Path(inspect.getfile(GPT)).read_bytes()).hexdigest(), torch=torch.__version__, stages={})
def checkpoint_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

for stage, filename in [('first', 'first_stage.pt'), ('second', 'second_stage.pt')]:
    torch.set_default_dtype(torch.float32)
    checkpoint_path = release['checkpoint_paths'][filename]
    checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
    model_args = checkpoint['model_args']
    config = GPTConfig(**model_args)
    state = {name.removeprefix('_orig_mod.'): tensor for name, tensor in checkpoint['model'].items()}
    with torch.device('meta'):
        model = GPT(config, speaker_emb_dim=checkpoint['meta']['speaker_emb_size'])
    model.load_state_dict(state, strict=True, assign=True)
    if config.target_vocab_sizes is None:
        for i in range(len(model.lm_heads)):
            model.lm_heads[i].weight = model.transformer.wtes[i].weight
    model.eval()
    tokenizer_args = checkpoint['meta']['tokenizer']
    tokenizer = TrainedBPETokeniser(**tokenizer_args)
    text = 'Hello.'
    text_ids = tokenizer.encode(text)
    speaker = torch.linspace(.02, 1.02, 256, dtype=torch.float32)
    speaker = (speaker / speaker.norm()).reshape(1, 1, 256)
    token_streams = [text_ids + [1, 1027]] if config.causal else [text_ids + [1, 2], [1024] * len(text_ids) + [3, 4]]
    tokens = torch.tensor([token_streams], dtype=torch.long)
    def run(speaker):
        capture = {}
        handle = model.transformer.ln_f.register_forward_hook(lambda _, inputs, output: capture.update(hidden=output))
        with torch.no_grad():
            logits, _ = model(tokens, speaker_embs=speaker)
        handle.remove()
        return dict(hidden=capture['hidden'].flatten().tolist(), logits=[value.flatten().tolist() for value in logits])
    original = run(speaker)
    model.double()
    torch.set_default_dtype(torch.float64)  # tok_emb zeros in original forward follow the default dtype.
    matched = run(speaker.double())
    examples = ['Hello, world!', "can't stop 123", '\t caf\u00e9 \u4f60\u597d', '  one\ntwo']
    result['stages'][stage] = dict(checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_hash(checkpoint_path),
        config=vars(config), state_tensor_count=len(state), unique_parameter_count=len(list(model.named_parameters())),
        text=text, text_ids=text_ids, token_streams=token_streams, speaker=speaker.flatten().tolist(),
        original_float32=original, matched_float64=matched,
        tokenizer=dict(name=tokenizer_args['name'], pattern=tokenizer_args['pat_str'],
            offset=tokenizer_args['offset'], special_tokens=tokenizer_args['special_tokens'],
            ranks=[dict(bytes=base64.b64encode(key).decode('ascii'), rank=value)
                   for key, value in sorted(tokenizer_args['mergeable_ranks'].items(), key=lambda pair: pair[1])]),
        tokenizer_examples=[dict(text=text, ids=tokenizer.encode(text)) for text in examples])
    print(json.dumps(dict(stage=stage, state_tensors=len(state), unique_parameters=len(list(model.named_parameters())),
        streams=len(token_streams), time=len(token_streams[0]))), flush=True)
    del model, state, checkpoint
    gc.collect()
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result), encoding='utf-8')
