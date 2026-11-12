"""Execute the authors' TorchMelSpectrogram and TacotronSTFT. Argument: TorToiSe checkout."""
import json
from pathlib import Path
import sys
import types
import torch

root = Path(sys.argv[1]).resolve()
package = types.ModuleType('tortoise')
package.__path__ = [str(root / 'tortoise')]
sys.modules['tortoise'] = package
from tortoise.models.arch_util import TorchMelSpectrogram
from tortoise.utils.audio import TacotronSTFT

torch.manual_seed(413)
ar = TorchMelSpectrogram().double().eval()
diffusion = TacotronSTFT(1024, 256, 1024, 100, 24000, 0, 12000).double().eval()
data = dict(source='neonbjb/tortoise-tts TorchMelSpectrogram, TacotronSTFT and data/mel_norms.pth', cases=[])
for name, audio in [('noise', torch.randn(1, 2048, dtype=torch.float64) * .15),
                    ('silence', torch.zeros(1, 2048, dtype=torch.float64))]:
    with torch.no_grad():
        ar_mel = ar(audio)
        diffusion_mel = diffusion.mel_spectrogram(audio)
    data['cases'].append(dict(name=name, audio=audio.flatten().tolist(),
        autoregressive=ar_mel.transpose(1, 2).flatten().tolist(), ar_frames=ar_mel.shape[-1],
        diffusion=diffusion_mel.transpose(1, 2).flatten().tolist(), diffusion_frames=diffusion_mel.shape[-1]))
out = Path(__file__).resolve().parents[2] / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/tortoise_mel_reference.json'
out.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
print(out)
