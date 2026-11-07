"""Small public-options contracts extracted from the independently executed checkpoints and official recipe."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--pretrained-reference', type=Path, required=True)
args = parser.parse_args()
reference = json.loads(args.pretrained_reference.read_text())
source = args.source / 'fam/llm/config/finetune_params.py'
keys = {'learning_rate', 'last_n_blocks_to_finetune', 'dropout', 'weight_decay', 'beta1', 'beta2', 'grad_clip'}
recipe = {}
for node in ast.parse(source.read_text()).body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in keys:
        recipe[node.targets[0].id] = ast.literal_eval(node.value)
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in keys:
        recipe[node.target.id] = ast.literal_eval(node.value)
assert recipe.keys() == keys
result = dict(repository=reference['repository'], revision=reference['revision'], source_revision=reference['source_revision'],
    stages={name: dict(config=stage['config'], tensors=stage['state_tensor_count']) for name, stage in reference['stages'].items()},
    fine_tuning=recipe, fine_tuning_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
destination = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/metavoice_release_contracts.json'
destination.write_text(json.dumps(result) + '\n', encoding='utf-8')
print(json.dumps(dict(stages=list(result['stages']), recipe=recipe)))
