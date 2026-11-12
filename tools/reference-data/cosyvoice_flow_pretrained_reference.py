"""Original CosyVoice-300M full flow checkpoint and ten-step prompted inference.

Arguments: pinned author checkout, Matcha checkout, original weights, output JSON.
No neural forward method is replaced. Optional serving imports are excluded;
the non-neural mask helper's dtype whitelist is relaxed to admit double. Original
float32 execution is retained. Double operator precision is explicitly isolated.
"""
import ast, json, logging, math, sys, types
from pathlib import Path
from unittest.mock import patch
import torch
from omegaconf import DictConfig
from torch.utils._python_dispatch import TorchDispatchMode

checkout, matcha, release, destination = map(Path, sys.argv[1:5])
sys.path[:0] = [str(checkout.resolve()), str(matcha.resolve())]; torch.set_num_threads(1)
registry_path = checkout/'cosyvoice/utils/class_utils.py'
registry_source = '\n'.join(line for line in registry_path.read_text(encoding='utf-8').splitlines()
    if not line.startswith(('from cosyvoice.llm', 'from cosyvoice.flow', 'from cosyvoice.hifigan', 'from cosyvoice.cli')))
registry = types.ModuleType('cosyvoice.utils.class_utils'); sys.modules[registry.__name__] = registry
exec(compile(registry_source.split('def get_model_type')[0], str(registry_path), 'exec'), registry.__dict__)
utils = types.ModuleType('matcha.utils'); utils.__path__ = [str(matcha/'matcha/utils')]
logger = types.ModuleType('matcha.utils.pylogger'); logger.get_pylogger = logging.getLogger
sys.modules['matcha.utils'] = utils; sys.modules['matcha.utils.pylogger'] = logger
import cosyvoice.utils.common as common
path = checkout/'cosyvoice/utils/common.py'
function = next(node for node in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(node, ast.FunctionDef) and node.name == 'mask_to_bias')
function.body = [node for node in function.body if not (isinstance(node, ast.Assert)
    and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == 'dtype')]
exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), 'exec'), common.__dict__)
from cosyvoice.transformer.encoder import ConformerEncoder
from cosyvoice.transformer.embedding import EspnetRelPositionalEncoding
from cosyvoice.flow.decoder import ConditionalDecoder
from cosyvoice.flow.flow_matching import ConditionalCFM
from cosyvoice.flow.flow import MaskedDiffWithXvec
from cosyvoice.flow.length_regulator import InterpolateRegulator

encoder = ConformerEncoder(512, output_size=512, attention_heads=8, linear_units=2048, num_blocks=6,
    dropout_rate=.1, positional_dropout_rate=.1, attention_dropout_rate=.1, normalize_before=True,
    input_layer='linear', pos_enc_layer_type='rel_pos_espnet', selfattention_layer_type='rel_selfattn',
    use_cnn_module=False, macaron_style=False)
estimator = ConditionalDecoder(320, 80, channels=[256, 256], dropout=0, attention_head_dim=64,
    n_blocks=4, num_mid_blocks=12, num_heads=8, act_fn='gelu')
cfm = ConditionalCFM(240, DictConfig(dict(sigma_min=1e-6, solver='euler', t_scheduler='cosine',
    training_cfg_rate=.2, inference_cfg_rate=.7, reg_loss_type='l1')), n_spks=1, spk_emb_dim=80, estimator=estimator)
model = MaskedDiffWithXvec(512, 80, 192, vocab_size=4096, input_frame_rate=50, only_mask_loss=True,
    encoder=encoder, length_regulator=InterpolateRegulator(80, sampling_ratios=[1, 1, 1, 1]), decoder=cfm).eval()
model.load_state_dict(torch.load(release/'flow.pt', weights_only=True, map_location='cpu'), strict=True)
flat = lambda value: value.detach().flatten().tolist()
text = torch.tensor([[1, 13, 42, 1200, 56, 77]]); prompt = torch.tensor([[3, 21, 100]])
prompt_mel = (torch.sin(torch.arange(400, dtype=torch.float64)*.013)*.09).reshape(1, 5, 80)
speaker = (torch.cos(torch.arange(192, dtype=torch.float64)*.13)*.17)[None]
noise = (torch.sin(torch.arange(1200, dtype=torch.float64)*.017)*.13).reshape(1, 80, 15)
data = dict(tokens=flat(text), prompt_tokens=flat(prompt), prompt_mel=flat(prompt_mel), speaker=flat(speaker), noise=flat(noise))

class DoubleOperators(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        dtype = kwargs.get('dtype') if func is torch.ops.aten._to_copy.default else args[1] if func is torch.ops.aten.to.dtype else None
        if dtype == torch.float32:
            if args[0].dtype == torch.float64: return args[0].clone()
            if func is torch.ops.aten._to_copy.default: kwargs = dict(kwargs, dtype=torch.float64)
            else: args = (args[0], torch.float64, *args[2:])
        return func(*args, **kwargs)

def execute(dtype):
    states = []; prepared = {}
    handles = [model.length_regulator.register_forward_hook(lambda m, a, out: prepared.update(mu=flat(out[0]), mu_shape=list(out[0].shape))),
               model.decoder.estimator.register_forward_hook(lambda m, a, out: states.append(flat(out)))]
    # inference invokes length_regulator.inference directly; its internal model
    # is observed separately through the final estimator's arguments.
    def capture(m, args, kwargs):
        prepared.update(mu=flat(kwargs['mu']), mu_shape=list(kwargs['mu'].shape), condition=flat(kwargs['cond']))
    handles.append(model.decoder.register_forward_pre_hook(capture, with_kwargs=True))
    try:
        with torch.inference_mode(), patch('torch.randn_like', side_effect=lambda x, **kw: noise.to(dtype)):
            output, cache = model.inference(text, torch.tensor([6]), prompt, torch.tensor([3]),
                prompt_mel.to(dtype), torch.tensor([5]), speaker.to(dtype), torch.zeros(1, 80, 0, 2, dtype=dtype))
        return dict(output=flat(output), output_shape=list(output.shape), estimator_steps=states, **prepared)
    finally:
        for handle in handles: handle.remove()

data['original_float32'] = execute(torch.float32)
model.double()
for module in model.modules():
    if isinstance(module, EspnetRelPositionalEncoding):
        half = (module.pe.shape[1]-1)//2
        position = torch.arange(half, -half-1, -1, dtype=torch.float64)[:, None]
        frequency = torch.exp(torch.arange(0, module.d_model, 2, dtype=torch.float64)*-(math.log(10000.0)/module.d_model))
        pe = torch.empty(position.shape[0], module.d_model, dtype=torch.float64)
        pe[:, 0::2] = torch.sin(position*frequency); pe[:, 1::2] = torch.cos(position*frequency); module.pe = pe[None]
torch.set_default_dtype(torch.float64)
with DoubleOperators(): data['float64_operators_derived_buffers'] = execute(torch.float64)
destination.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
print('Generated complete trained CosyVoice-300M flow, ten original Euler steps and prompted output.', flush=True)
