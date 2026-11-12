"""Original UniAudio mel critic, STFT/PQMF reconstruction and GAN objective gradients.

Arguments: pinned UniAudio checkout. Small independent fixtures are written to tests.
Compatibility only: scipy.signal.kaiser's moved name and torch.stft's old real/imag
return layout. An explicitly labeled double-operator context isolates original
float32 casts without replacing author neural forward methods.
"""
import json
from pathlib import Path
import sys
import torch
from torch.utils._python_dispatch import TorchDispatchMode
import scipy.signal
from omegaconf import OmegaConf

checkout=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(checkout/'codec'))
scipy.signal.kaiser=scipy.signal.windows.kaiser
original_stft=torch.stft
def legacy_stft(*args,**kwargs):
    kwargs.setdefault('return_complex',False)
    return original_stft(*args,**kwargs)
torch.stft=legacy_stft
from modules.discriminators.frequency_discriminator import FrequenceDiscriminator,MultiFrequencyDiscriminator
from modules.commons.torch_stft import TorchSTFT
from losses.generator_loss import GeneratorSTFTLoss
from losses.discriminator_loss import MSEDiscriminatorLoss

class DoubleOperators(TorchDispatchMode):
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        dtype=kwargs.get('dtype') if func is torch.ops.aten._to_copy.default else \
            args[1] if func is torch.ops.aten.to.dtype else None
        if args and isinstance(args[0],torch.Tensor) and args[0].dtype==torch.float64 and dtype==torch.float32:
            # Preserve the copy's fresh-tensor contract for autograd while isolating dtype.
            return args[0].clone()
        return func(*args,**kwargs)

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(231000704)
flat=lambda value:value.detach().flatten().tolist()
record=lambda module:dict(weights={name:flat(value) for name,value in module.state_dict().items()},
                          shapes={name:list(value.shape) for name,value in module.state_dict().items()})
root=Path(__file__).resolve().parents[2]/'tests/AiDotNet.Tests/TextToSpeech/ReferenceData'
data=dict(source='Original codec/modules/discriminators/frequency_discriminator.py and codec/losses; double operator precision')

with DoubleOperators():
    critic=FrequenceDiscriminator(2,64).double()
    features=torch.randn(1,2,17,33,requires_grad=True)
    score,maps=critic(features)
    loss=score.square().mean()+sum(value.square().mean()*.03 for value in maps)
    loss.backward()
    data['critic']=dict(**record(critic),input=flat(features),input_shape=list(features.shape),score=flat(score),
        score_shape=list(score.shape),features=[flat(value) for value in maps],feature_shapes=[list(value.shape) for value in maps],
        loss=loss.item(),input_gradient=flat(features.grad),
        gradients={name:flat(value.grad) for name,value in critic.named_parameters()})

    frontend=TorchSTFT(16,4,16,normalized=True,domain='double',mel_scale=True,sample_rate=16000).double()
    audio=(.13*torch.sin(torch.arange(128)*.11)+.07*torch.cos(torch.arange(128)*.041)).requires_grad_()
    magnitude,_=frontend.transform(audio[None])
    loss=(magnitude.square()).mean()
    loss.backward()
    data['frontend']=dict(audio=flat(audio),mel_basis=flat(frontend.mel_scale.fb),output=flat(magnitude),
        output_shape=list(magnitude.shape),loss=loss.item(),input_gradient=flat(audio.grad))

    configuration=OmegaConf.create(dict(adv_criterion='MSEGLoss',use_feature_match=True,feat_match_loss_weight=20,
        use_mel_loss=False,use_full_stft_loss=True,use_sub_stft_loss=True,
        full_stft_loss_weight=1,sub_stft_loss_weight=1,
        full_multi_scale_stft_loss=dict(fft_sizes=[32,64],win_sizes=[24,48],hop_sizes=[8,16]),
        sub_multi_scale_stft_loss=dict(num_bands=6,fft_sizes=[8,16],win_sizes=[6,12],hop_sizes=[2,4])))
    criterion=GeneratorSTFTLoss(configuration).double()
    real=.15*torch.sin(torch.arange(480)*.09)+.04*torch.cos(torch.arange(480)*.023)
    fake=(real*.91+.013*torch.sin(torch.arange(480)*.17)).clone().requires_grad_()
    real_scores=[torch.randn(1,1,2,3),torch.randn(1,1,3,2)]
    fake_scores=[torch.randn(1,1,2,3,requires_grad=True),torch.randn(1,1,3,2,requires_grad=True)]
    real_maps=[[torch.randn(1,2,3,4),torch.randn(1,3,2,3)] for _ in range(2)]
    fake_maps=[[torch.randn(1,2,3,4,requires_grad=True),torch.randn(1,3,2,3,requires_grad=True)] for _ in range(2)]
    generator,items=criterion(real[None,None],fake[None,None],dict(mfd=real_scores),dict(mfd=fake_scores),
        dict(mfd=real_maps),dict(mfd=fake_maps))
    generator.backward()
    reconstruction,_=criterion(real[None,None],fake.detach()[None,None],{}, {}, {}, {},use_adv_loss=False)
    discriminator=MSEDiscriminatorLoss()(real_scores,fake_scores)
    discriminator_gradients=torch.autograd.grad(discriminator,fake_scores)
    data['objective']=dict(real=flat(real),fake=flat(fake),generator_loss=generator.item(),
        reconstruction_loss=reconstruction.item(),discriminator_loss=discriminator.item(),items=items,
        fake_gradient=flat(fake.grad),real_scores=[flat(x) for x in real_scores],
        fake_scores=[flat(x) for x in fake_scores],score_shapes=[list(x.shape) for x in fake_scores],
        real_features=[[flat(x) for x in group] for group in real_maps],
        fake_features=[[flat(x) for x in group] for group in fake_maps],
        feature_shapes=[[list(x.shape) for x in group] for group in fake_maps],
        score_gradients=[flat(x.grad) for x in fake_scores],
        feature_gradients=[[flat(x.grad) for x in group] for group in fake_maps],
        discriminator_score_gradients=[flat(x) for x in discriminator_gradients])

parameter=torch.nn.Parameter(torch.tensor(1.0))
optimizer=torch.optim.AdamW([parameter],lr=2e-4,betas=(.8,.99),eps=1e-6,weight_decay=.01)
scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=.999)
rates=[optimizer.param_groups[0]['lr']]
for _ in range(3):
    optimizer.step();scheduler.step();rates.append(optimizer.param_groups[0]['lr'])
data['optimizer']=dict(epoch_rates=rates,betas=optimizer.param_groups[0]['betas'],
                       epsilon=optimizer.param_groups[0]['eps'],weight_decay=optimizer.param_groups[0]['weight_decay'])
(root/'uniaudio_codec_training_reference.json').write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print('Original 7-layer critic, frontend, full/sub-band STFT and GAN objectives with all gradients written.',flush=True)
