using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// #1714: back in the default gate. The HeavyTimeout deferral here was measured away: in the
// nightly lane (run 38117342722, one class per process under a managed heap cap of 70% of
// runner RAM) this class ran 13 of 13 tests green, slowest 5.5 s and no OutOfMemoryException
// -- far inside the 120 s per-test gate. Re-defer only against a fresh measurement.
public class DiffWaveModelTests : DiffusionModelTestBase<float>
{
    // DiffWave operates on raw 1D audio waveforms — Kong et al. 2020 §3 /
    // Figure 1 — NOT on image-like 4D latents. The previous [1, 4, 64, 64]
    // scaffold default tripped the [B, C, T] + [B, C] broadcast-add in
    // DiffWaveResidualBlock (PR #1408 Diffusion S-Z shard:
    // "Tensor shapes must match. Got [1, 4, 64, 64] and [1, 64]").
    // [B=1, C=1 (mono), T=256] is a small but paper-faithful test shape that
    // keeps invariants meaningful while staying fast.
    protected override int[] InputShape => [1, 1, 256];
    protected override int[] OutputShape => [1, 1, 256];

    protected override IDiffusionModel<float> CreateModel()
        => new DiffWaveModel<float>(seed: 42);
}
